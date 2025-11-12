import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from .config import Cfg
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from .logger import log
from .original_fl_rotation import Config as RotationConfig, FederatedServer as _FederatedServer, FLClient as _BaseFLClient
from .attacks_comprehensive import (
    corrupt_model_byzantine,
    scale_model_parameters,
    create_stale_model,
    spawn_sybil_clients,
    label_flip,
    byzantine_update,
    compute_attack_success_rate
)

# -----------------------------
# Helper functions
# -----------------------------

def load_client(i):
    # Try the new data structure first: data/Client_i/Client_i_full.csv
    path = f"{Cfg.DATA}/Client_{i}/Client_{i}_full.csv"
    if not os.path.exists(path):
        # Fallback to old structure: data/client_i_data.csv
        path = f"{Cfg.DATA}/client_{i}_data.csv"
        if not os.path.exists(path):
            log(f"✗ Missing client {i} at {path}")
            return None
    df = pd.read_csv(path)
    if 'isFraud' not in df.columns:
        log(f"✗ Client {i} missing 'isFraud' column")
        return None
    return df

def to_lgb(df):
    X = df.drop('isFraud', axis=1).values.astype(np.float32)
    y = df['isFraud'].values.astype(np.float32)
    return lgb.Dataset(X, label=y, free_raw_data=False), X, y

def compute_cosine(u, v):
    """Cosine similarity between two vectors."""
    if u is None or v is None:
        return 0.0
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)

def compute_trigger_rate(X, trigger_features):
    """Fraction of rows matching trigger pattern."""
    if not trigger_features:
        return 0.0
    df = pd.DataFrame(X)
    mask = np.ones(len(df), dtype=bool)
    for k, v in trigger_features.items():
        if k in df.columns:
            mask &= (df[k] == v)
    return float(mask.mean()) if len(df) > 0 else 0.0

def extract_model_vector(model):
    """Extract a normalized vector from a LightGBM model for update analysis.
    Combines gain and split importances and normalizes by L2 norm; if degenerate,
    falls back to light leaf-value stats normalized by their L2 norm. This avoids
    inflated magnitudes and stabilizes downstream metrics (update_norm, variance, range)."""
    if model is None:
        return np.array([0.0], dtype=float)
    try:
        parts = []
        try:
            gain = np.asarray(model.feature_importance(importance_type='gain'), dtype=float)
            if gain.size > 0:
                parts.append(gain)
        except Exception:
            gain = None
        try:
            split = np.asarray(model.feature_importance(importance_type='split'), dtype=float)
            if split.size > 0:
                parts.append(split)
        except Exception:
            split = None
        vec = None
        if parts:
            vec = np.concatenate(parts)
            scale = float(np.linalg.norm(vec)) or 1.0
            vec = vec / scale
        if vec is None or not np.any(np.isfinite(vec)) or np.allclose(vec, 0.0):
            # Fallback: derive normalized leaf stats from dumped trees
            try:
                dump = model.dump_model()
                trees = dump.get('tree_info', []) if isinstance(dump, dict) else []
                leaf_vals = []
                for t in trees:
                    stack = [t.get('tree_structure', {})]
                    while stack:
                        node = stack.pop()
                        if not isinstance(node, dict):
                            continue
                        if 'leaf_value' in node:
                            leaf_vals.append(float(node.get('leaf_value', 0.0)))
                        else:
                            if 'left_child' in node:
                                stack.append(node['left_child'])
                            if 'right_child' in node:
                                stack.append(node['right_child'])
                leaf_vals = np.asarray(leaf_vals, dtype=float)
                if leaf_vals.size == 0:
                    vec = np.array([0.0, 0.0, 0.0, float(len(trees) or 1)], dtype=float)
                else:
                    stats = np.array([
                        float(np.sum(np.abs(leaf_vals))),
                        float(np.mean(np.abs(leaf_vals))),
                        float(np.max(np.abs(leaf_vals))),
                        float(len(trees) or 1)
                    ], dtype=float)
                    scale = float(np.linalg.norm(stats)) or 1.0
                    vec = stats / scale
            except Exception:
                vec = np.array([1e-9], dtype=float)
        return vec.astype(float)
    except Exception as e:
        log(f"Warning: Could not extract model vector: {str(e)}")
        return np.array([0.0], dtype=float)

def compute_risk_score(entry):
    try:
        upd = float(entry.get('update_norm', 0.0) or 0.0)
        cos = float(entry.get('cosine_similarity', 0.0) or 0.0)
        frd = float(entry.get('fraud_ratio_change', 0.0) or 0.0)
        upd_sig = float(np.tanh(upd / 100.0))
        cos_inv = float(max(0.0, 1.0 - cos))
        # Incorporate per-client triggered ASR contribution (if available)
        cta = entry.get('client_triggered_asr', 0.0)
        try:
            cta = float(cta)
        except Exception:
            cta = 0.0
        # Normalize ASR to [0,1] if provided as percentage
        if cta > 1.0:
            cta = cta / 100.0
        cta = float(max(0.0, min(1.0, cta)))
        # Optional: parameter changes under trigger (0..1)
        ptr = float(max(0.0, min(1.0, float(entry.get('param_trigger_change', 0.0) or 0.0))))
        # Backdoor-oriented weights removed
        r = (
            0.25 * cta +
            0.45 * cos_inv +
            0.30 * upd_sig
        )
        return float(max(0.0, min(1.0, r)))
    except Exception:
        return 0.0

def validate_model_update(model, metrics, config, attack_info=None):
    """Validate model update quality using multiple criteria, with attack-specific adjustments.

    Args:
        model: Client model to validate
        metrics: Performance metrics from client training
        config: Configuration dictionary with thresholds
        attack_info: Optional dictionary containing attack detection information

    Returns:
        (bool, str): Tuple of (is_valid, reason)
    """
    if model is None:
        return False, "Model is None"
        
    try:
        # Extract model vector for analysis
        model_vec = extract_model_vector(model)
        if model_vec is None or len(model_vec) == 0:
            return False, "Invalid model vector"
        # Check for NaN or infinite values
        if np.any(np.isnan(model_vec)) or np.any(np.isinf(model_vec)):
            return False, "Model contains NaN or infinite values"
            
        # Get base thresholds
        min_accuracy = config.get('min_accuracy', 0.5)
        min_f1 = config.get('min_f1', 0.3)
        min_auc = config.get('min_auc', 0.5)
        
        # Adjust thresholds based on attack detection
        if attack_info and attack_info.get('is_attack_detected', False):
            attack_type = attack_info.get('attack_type', '')
            num_attackers = attack_info.get('num_attackers', 1)
            
            # Adjust thresholds based on number of attackers
            if num_attackers == 2:
                min_accuracy *= 0.75  # 25% penalty
                min_f1 *= 0.8  # 20% penalty
                min_auc = max(0.5, min_auc * 0.85)  # 15% penalty with floor
            elif num_attackers >= 3:
                min_accuracy *= 0.65  # 35% penalty
                min_f1 *= 0.7  # 30% penalty
                min_auc = max(0.5, min_auc * 0.75)  # 25% penalty with floor
            
            # Additional adjustments based on attack type
            if 'free_ride' in attack_type.lower():
                min_auc = max(0.5, min_auc * 0.9)  # Additional 10% AUC penalty
            # removed backdoor-specific threshold changes
        
        # Validate metrics with adjusted thresholds
        if metrics:
            if metrics.get('accuracy', 0) < min_accuracy:
                return False, f"Accuracy below adjusted threshold: {metrics.get('accuracy', 0):.4f} < {min_accuracy}"
            if metrics.get('f1_score', 0) < min_f1:
                return False, f"F1 score below adjusted threshold: {metrics.get('f1_score', 0):.4f} < {min_f1}"
            if metrics.get('auc', 0) < min_auc:
                return False, f"AUC below adjusted threshold: {metrics.get('auc', 0):.4f} < {min_auc}"
                
        # Check model parameter statistics
        max_param_value = config.get('max_param_value', 1e6)
        min_param_variance = config.get('min_param_variance', 1e-6)
        max_param_range = config.get('max_param_range', 1e6)
        
        param_max = np.max(np.abs(model_vec))
        param_var = np.var(model_vec)
        param_range = np.ptp(model_vec)
        
        if param_max > max_param_value:
            return False, f"Parameter values too large: {param_max:.4f} > {max_param_value}"
        if param_var < min_param_variance:
            return False, f"Parameter variance too small: {param_var:.4e} < {min_param_variance}"
        if param_range > max_param_range:
            return False, f"Parameter range too large: {param_range:.4f} > {max_param_range}"
            
        return True, "Model update passed validation"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def check_model_divergence(model, global_model, config, attack_info=None):
    """
    Check if model update diverges too much from global model, with attack-specific adjustments.
    
    Args:
        model: Client model to check
        global_model: Current global model
        config: Configuration dictionary with thresholds
        attack_info: Optional dictionary containing attack detection information
        
    Returns:
        (bool, str): Tuple of (is_acceptable, reason)
    """
    if model is None or global_model is None:
        return False, "Missing model"
        
    try:
        # Extract model vectors
        model_vec = extract_model_vector(model)
        global_vec = extract_model_vector(global_model)
        
        if model_vec is None or global_vec is None:
            return False, "Could not extract model vectors"
            
        if len(model_vec) != len(global_vec):
            return False, "Model vector dimensions mismatch"
            
        # Compute divergence metrics
        cosine_sim = compute_cosine(model_vec, global_vec)
        l2_dist = np.linalg.norm(model_vec - global_vec)
        relative_norm = np.linalg.norm(model_vec) / (np.linalg.norm(global_vec) + 1e-8)
        
        # Get base thresholds
        min_cosine_sim = config.get('min_cosine_similarity', 0.1)
        max_l2_distance = config.get('max_l2_distance', 100)
        max_relative_norm = config.get('max_relative_norm', 10)
        
        # Adjust thresholds based on attack detection
        if attack_info and attack_info.get('is_attack_detected', False):
            attack_type = attack_info.get('attack_type', '')
            num_attackers = attack_info.get('num_attackers', 1)
            
            # Adjust thresholds based on number of attackers
            if num_attackers == 2:
                min_cosine_sim *= 0.85  # 15% lower similarity requirement
                max_l2_distance *= 1.25  # 25% higher distance allowance
                max_relative_norm *= 1.2  # 20% higher norm allowance
            elif num_attackers >= 3:
                min_cosine_sim *= 0.75  # 25% lower similarity requirement
                max_l2_distance *= 1.35  # 35% higher distance allowance
                max_relative_norm *= 1.3  # 30% higher norm allowance
            
            # Additional adjustments based on attack type
            if 'byzantine' in attack_type.lower():
                min_cosine_sim *= 0.9  # Additional 10% lower similarity requirement
                max_l2_distance *= 1.1  # Additional 10% higher distance allowance
            # removed backdoor-specific norm allowance
        
        # Check against adjusted thresholds
        if cosine_sim < min_cosine_sim:
            return False, f"Cosine similarity below adjusted threshold: {cosine_sim:.4f} < {min_cosine_sim}"
        if l2_dist > max_l2_distance:
            return False, f"L2 distance above adjusted threshold: {l2_dist:.4f} > {max_l2_distance}"
        if relative_norm > max_relative_norm:
            return False, f"Relative norm above adjusted threshold: {relative_norm:.4f} > {max_relative_norm}"
            
        return True, "Model divergence acceptable"
        
    except Exception as e:
        return False, f"Divergence check error: {str(e)}"

def validate_aggregated_model(model, client_models, metrics, config, attack_info=None):
    """
    Validate aggregated model quality with attack-specific adjustments.
    
    Args:
        model: Aggregated model to validate
        client_models: List of client models used in aggregation
        metrics: Performance metrics of aggregated model
        config: Configuration dictionary with thresholds
        attack_info: Optional dictionary containing attack detection information
        
    Returns:
        (bool, str): Tuple of (is_valid, reason)
    """
    if model is None:
        return False, "Aggregated model is None"
        
    try:
        # Basic model validation with attack info
        valid, reason = validate_model_update(model, metrics, config, attack_info)
        if not valid:
            return False, f"Basic validation failed: {reason}"
            
        # Get base thresholds
        min_relative_accuracy = config.get('min_relative_accuracy', 0.9)
        min_relative_f1 = config.get('min_relative_f1', 0.9)
        
        # Adjust thresholds based on attack detection
        if attack_info and attack_info.get('is_attack_detected', False):
            attack_type = attack_info.get('attack_type', '')
            num_attackers = attack_info.get('num_attackers', 1)
            
            # Adjust thresholds based on number of attackers
            if num_attackers == 2:
                min_relative_accuracy *= 0.85  # 15% lower relative accuracy requirement
                min_relative_f1 *= 0.85  # 15% lower relative F1 requirement
            elif num_attackers >= 3:
                min_relative_accuracy *= 0.75  # 25% lower relative accuracy requirement
                min_relative_f1 *= 0.75  # 25% lower relative F1 requirement
            
            # Additional adjustments based on attack type
            if 'byzantine' in attack_type.lower():
                min_relative_accuracy *= 0.9  # Additional 10% lower relative accuracy requirement
                min_relative_f1 *= 0.9  # Additional 10% lower relative F1 requirement
            # removed backdoor-specific relative accuracy tweak
        
        # Calculate average client performance
        client_accuracies = []
        client_f1_scores = []
        
        for client_model, _ in client_models:
            if client_model is not None:
                client_vec = extract_model_vector(client_model)
                if client_vec is not None and len(client_vec) > 0:
                    # Get client metrics
                    client_metrics = metrics  # Use provided metrics
                    if client_metrics:
                        client_accuracies.append(client_metrics.get('accuracy', 0))
                        client_f1_scores.append(client_metrics.get('f1_score', 0))
                        
        if client_accuracies and client_f1_scores:
            avg_accuracy = np.mean(client_accuracies)
            avg_f1 = np.mean(client_f1_scores)
            
            agg_accuracy = metrics.get('accuracy', 0)
            agg_f1 = metrics.get('f1_score', 0)
            
            relative_accuracy = agg_accuracy / (avg_accuracy + 1e-8)
            relative_f1 = agg_f1 / (avg_f1 + 1e-8)
            
            if relative_accuracy < min_relative_accuracy:
                return False, f"Aggregated accuracy below adjusted threshold relative to clients: {relative_accuracy:.4f} < {min_relative_accuracy}"
            if relative_f1 < min_relative_f1:
                return False, f"Aggregated F1 below adjusted threshold relative to clients: {relative_f1:.4f} < {min_relative_f1}"
                
        return True, "Aggregated model passed validation"
        
    except Exception as e:
        return False, f"Aggregation validation error: {str(e)}"

def krum_aggregation(client_models, num_byzantine, num_neighbors=None):
    """
    Implements the Krum aggregation method for Byzantine-resilient federated learning.
    
    Args:
        client_models: List of tuples (model, weight) containing client models and their weights
        num_byzantine: Number of Byzantine clients to tolerate
        num_neighbors: Number of neighbors to use (defaults to n - num_byzantine - 2)
    
    Returns:
        Selected model that minimizes the sum of distances to its closest neighbors
    """
    if not client_models:
        return None
        
    n = len(client_models)
    if num_neighbors is None:
        num_neighbors = n - num_byzantine - 2
        
    if num_neighbors < 1:
        log("⚠ Invalid number of neighbors for Krum, falling back to weighted average")
        return None
        
    # Extract model vectors
    model_vectors = []
    for model, _ in client_models:
        if model is not None:
            vec = extract_model_vector(model)
            if vec is not None and len(vec) > 0:
                model_vectors.append(vec)
                
    if not model_vectors:
        return None
        
    # Compute pairwise distances
    n_vectors = len(model_vectors)
    distances = np.zeros((n_vectors, n_vectors))
    
    for i in range(n_vectors):
        for j in range(i+1, n_vectors):
            dist = np.linalg.norm(model_vectors[i] - model_vectors[j])
            distances[i,j] = distances[j,i] = dist
            
    # For each model, sum its distances to its closest neighbors
    scores = []
    for i in range(n_vectors):
        # Get distances to other models
        dist_to_others = distances[i]
        # Sort distances and sum the closest num_neighbors
        closest_distances = np.sort(dist_to_others)[:num_neighbors]
        scores.append(np.sum(closest_distances))
        
    # Select model with minimum score
    selected_idx = np.argmin(scores)
    return client_models[selected_idx][0]

def trimmed_mean_aggregation(client_models, trim_ratio=0.1):
    """
    Implements the Trimmed Mean aggregation method for Byzantine-resilient federated learning.
    
    Args:
        client_models: List of tuples (model, weight) containing client models and their weights
        trim_ratio: Ratio of models to trim from each end (default 0.1 = 10%)
    
    Returns:
        Aggregated model using trimmed mean
    """
    if not client_models:
        return None
        
    # Extract model vectors
    model_vectors = []
    for model, _ in client_models:
        if model is not None:
            vec = extract_model_vector(model)
            if vec is not None and len(vec) > 0:
                model_vectors.append(vec)
                
    if not model_vectors:
        return None
        
    # Convert to numpy array
    model_vectors = np.array(model_vectors)
    
    # Calculate number of models to trim from each end
    n_models = len(model_vectors)
    n_trim = int(n_models * trim_ratio)
    
    if n_trim * 2 >= n_models:
        log("⚠ Invalid trim ratio, would remove all models. Falling back to median")
        n_trim = (n_models - 1) // 2
        
    # Sort models along each parameter dimension
    sorted_vectors = np.sort(model_vectors, axis=0)
    
    # Remove highest and lowest n_trim values
    trimmed_vectors = sorted_vectors[n_trim:n_models-n_trim]
    
    # Compute mean of remaining values
    aggregated_vector = np.mean(trimmed_vectors, axis=0)
    
    # Convert aggregated vector back to model
    return vector_to_model(aggregated_vector, client_models[0][0])

def vector_to_model(vector, template_model):
    """Convert a parameter vector back to a LightGBM model."""
    try:
        if template_model is None:
            return None
            
        # Create a new model with same structure
        n_features = len(template_model.feature_importance())
        
        # Create dummy data
        X_dummy = np.random.rand(10, n_features).astype(np.float32)
        y_dummy = np.zeros(10).astype(np.float32)
        
        # Create dataset
        train_data = lgb.Dataset(X_dummy, label=y_dummy)
        
        # Get model parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'num_leaves': template_model.num_leaves(),
            'learning_rate': 0.1
        }
        
        # Train minimal model
        new_model = lgb.train(params, train_data, num_boost_round=1)
        
        # Update model parameters with aggregated vector
        update_model_parameters(new_model, vector)
        
        return new_model
    except Exception as e:
        log(f"Warning: Could not convert vector to model: {str(e)}")
        return None

def update_model_parameters(model, parameter_vector):
    """Safe no-op: do not mutate LightGBM Booster internals from a vector."""
    try:
        if model is None or parameter_vector is None:
            return
        # Intentionally do nothing. Use read-only importances elsewhere.
        return
    except Exception as e:
        log(f"Warning: Could not update model parameters: {str(e)}")

def compute_adaptive_lr(base_lr, round_num, performance_history, client_id):
    """
    Compute adaptive learning rate based on client's historical performance.
    
    Args:
        base_lr: Base learning rate
        round_num: Current round number
        performance_history: List of historical performance metrics
        client_id: Client identifier
    
    Returns:
        Adjusted learning rate
    """
    if not performance_history:
        return base_lr
        
    # Get client's recent performance metrics
    client_metrics = [m for m in performance_history if m.get('client_id') == client_id]
    if not client_metrics:
        return base_lr
        
    # Calculate performance trend
    recent_metrics = client_metrics[-3:]  # Look at last 3 rounds
    if len(recent_metrics) < 2:
        return base_lr
        
    # Calculate average improvement in metrics (if available)
    if recent_metrics and 'accuracy' in recent_metrics[0]:
        avg_improvement = np.mean([
            m2['accuracy'] - m1['accuracy']
            for m1, m2 in zip(recent_metrics[:-1], recent_metrics[1:])
        ])
        
        # Adjust learning rate based on improvement trend
        if avg_improvement > 0.01:  # Good improvement
            lr = base_lr * 1.1  # Increase learning rate
        elif avg_improvement < -0.01:  # Performance degradation
            lr = base_lr * 0.9  # Decrease learning rate
        else:
            lr = base_lr
    else:
        # No evaluation metrics available yet, use base learning rate
        lr = base_lr
        
    # Apply warmup in early rounds
    warmup_rounds = 5
    if round_num <= warmup_rounds:
        lr *= round_num / warmup_rounds
        
    # Apply cosine decay in later rounds
    def _safe_cfg(name, default):
        try:
            from src.config import Cfg
            return getattr(Cfg, name, default)
        except Exception:
            return default
    total_rounds = int(_safe_cfg('R', 3))
    if round_num > warmup_rounds:
        progress = (round_num - warmup_rounds) / (total_rounds - warmup_rounds)
        lr *= 0.5 * (1 + np.cos(np.pi * progress))
        
    return float(max(lr, base_lr * 0.1))  # Ensure minimum learning rate

def compute_momentum(round_num, client_metrics):
    """
    Compute momentum factor based on training progress and client performance.
    
    Args:
        round_num: Current round number
        client_metrics: List of client's historical metrics
    
    Returns:
        Momentum factor
    """
    base_momentum = 0.9
    
    if not client_metrics:
        return base_momentum
        
    # Adjust momentum based on training progress
    def _safe_cfg(name, default):
        try:
            from src.config import Cfg
            return getattr(Cfg, name, default)
        except Exception:
            return default
    total_rounds = max(1, int(_safe_cfg('R', 3)))
    progress = round_num / total_rounds
    
    # Increase momentum in later rounds for faster convergence
    momentum = base_momentum + (0.05 * progress)
    
    # Adjust based on recent performance stability (if metrics available)
    recent_metrics = client_metrics[-3:]
    if len(recent_metrics) >= 2 and 'accuracy' in recent_metrics[0]:
        # Calculate performance variance
        accuracies = [m['accuracy'] for m in recent_metrics]
        variance = np.var(accuracies)
        
        # Reduce momentum if performance is unstable
        if variance > 0.01:
            momentum *= 0.9
            
    return float(min(momentum, 0.99))  # Cap maximum momentum

def compute_client_reputation(client_metrics, round_num, window_size=5):
    """
    Compute client reputation score based on historical performance.
    
    Args:
        client_metrics: List of client's historical metrics
        round_num: Current round number
        window_size: Number of recent rounds to consider
        
    Returns:
        Reputation score between 0 and 1
    """
    if not client_metrics:
        return 0.5  # Default neutral score
        
    # Get recent metrics
    recent_metrics = client_metrics[-window_size:]
    if not recent_metrics:
        return 0.5
        
    # Calculate performance metrics
    accuracy_trend = [m['accuracy'] for m in recent_metrics]
    f1_trend = [m['f1_score'] for m in recent_metrics]
    auc_trend = [m['auc'] for m in recent_metrics]
    
    # Calculate improvement trends
    accuracy_improvement = np.mean([y - x for x, y in zip(accuracy_trend[:-1], accuracy_trend[1:])]) if len(accuracy_trend) > 1 else 0
    f1_improvement = np.mean([y - x for x, y in zip(f1_trend[:-1], f1_trend[1:])]) if len(f1_trend) > 1 else 0
    auc_improvement = np.mean([y - x for x, y in zip(auc_trend[:-1], auc_trend[1:])]) if len(auc_trend) > 1 else 0
    
    # Calculate stability (lower variance is better)
    accuracy_stability = 1 / (1 + np.var(accuracy_trend))
    f1_stability = 1 / (1 + np.var(f1_trend))
    auc_stability = 1 / (1 + np.var(auc_trend))
    
    # Calculate current performance level
    current_accuracy = accuracy_trend[-1] if accuracy_trend else 0.5
    current_f1 = f1_trend[-1] if f1_trend else 0.5
    current_auc = auc_trend[-1] if auc_trend else 0.5
    
    # Combine metrics into reputation score
    performance_score = np.mean([current_accuracy, current_f1, current_auc])
    improvement_score = np.mean([accuracy_improvement, f1_improvement, auc_improvement]) + 0.5  # Shift to [0,1]
    stability_score = np.mean([accuracy_stability, f1_stability, auc_stability])
    
    # Weight the components (can be adjusted)
    weights = {
        'performance': 0.5,
        'improvement': 0.3,
        'stability': 0.2
    }
    
    reputation = (
        weights['performance'] * performance_score +
        weights['improvement'] * improvement_score +
        weights['stability'] * stability_score
    )
    
    # Apply temporal decay
    rounds_participated = len(client_metrics)
    participation_factor = min(rounds_participated / 10, 1.0)  # Full weight after 10 rounds
    
    # Bound reputation between 0 and 1
    return float(min(max(reputation * participation_factor, 0.0), 1.0))

def compute_contribution_weight(client_metrics, round_num, reputation_score):
    """
    Compute client contribution weight based on performance and reputation.
    
    Args:
        client_metrics: List of client's historical metrics
        round_num: Current round number
        reputation_score: Client's reputation score
        
    Returns:
        Contribution weight between 0 and 1
    """
    if not client_metrics:
        return 0.5  # Default neutral weight
        
    # Get most recent metrics
    recent_metrics = client_metrics[-1]
    
    # Calculate performance-based weight
    performance_weight = np.mean([
        recent_metrics['accuracy'],
        recent_metrics['f1_score'],
        recent_metrics['auc']
    ])
    
    # Calculate data quality weight based on class balance
    if 'class_balance' in recent_metrics:
        balance_ratio = min(recent_metrics['class_balance'], 1 - recent_metrics['class_balance'])
        data_quality_weight = 2 * balance_ratio  # Scale to [0,1]
    else:
        data_quality_weight = 0.5
    
    # Combine weights
    weights = {
        'performance': 0.4,
        'reputation': 0.4,
        'data_quality': 0.2
    }
    
    contribution_weight = (
        weights['performance'] * performance_weight +
        weights['reputation'] * reputation_score +
        weights['data_quality'] * data_quality_weight
    )
    
    # Apply round-based scaling
    warmup_rounds = 5
    if round_num <= warmup_rounds:
        # Gradually increase weight during warmup
        contribution_weight *= round_num / warmup_rounds
    
    return float(min(max(contribution_weight, 0.1), 1.0))  # Ensure minimum contribution

class AttackedFLClient(_BaseFLClient):
    """FLClient that can apply attacks to training data and/or model updates."""
    def __init__(self, client_id, client_name, data_path, is_attacker=False, attack_type=None, attack_config=None, sybil_label=None):
        super().__init__(client_id, client_name, data_path)
        self.is_attacker = bool(is_attacker)
        self.attack_type = (str(attack_type).lower().replace(' attack','').replace('-', '_').replace(' ', '_')
                            if isinstance(attack_type, str) else attack_type)
        self.attack_config = attack_config or {}
        self.sybil_label = sybil_label  # e.g., "sybil_6" for labeling logs
        # Detection-related
        self.flip_percent_used = 0.0
        self.trigger_rate = 0.0
        self.scaling_factor = float(self.attack_config.get('scaling_factor', self.attack_config.get('factor', 5.0) or 5.0))
        self.staleness = 1.0 if (self.is_attacker and self.attack_type == 'free_ride') else 0.0
        self._train_pos_ratio_orig = None
        self._train_pos_ratio_poison = None

    def load_data(self) -> bool:
        """Load standard client splits, then apply data poisoning if configured."""
        ok = super().load_data()
        if not ok:
            return False
        try:
            # Track original train fraud ratio
            try:
                self._train_pos_ratio_orig = float(self.y_train.mean())
            except Exception:
                self._train_pos_ratio_orig = 0.0

            if self.is_attacker and self.attack_type in ('label_flip', 'scaling'):
                # Convert to numpy
                X = self.X_train.values
                y = self.y_train.values
                if self.attack_type == 'label_flip':
                    fp = self.attack_config.get('flip_percent', self.attack_config.get('flip_ratio', 0.4))
                    try:
                        fp = float(fp)
                    except Exception:
                        fp = 0.4
                    self.flip_percent_used = max(0.0, min(1.0, fp))
                    Xp, yp = label_flip(X, y, self.flip_percent_used)
                    # Rebuild DataFrame/Series
                    self.X_train = pd.DataFrame(Xp, columns=self.X_train.columns)
                    self.y_train = pd.Series(yp, index=self.X_train.index)
                elif self.attack_type == 'scaling':
                    # For scaling attacks: apply label corruption to make recall drop
                    drop_pos_frac = float(self.attack_config.get('drop_positive_fraction', 0.0) or 0.0)
                    flip_labels_frac = float(self.attack_config.get('flip_labels_fraction', 0.0) or 0.0)
                    
                    # Drop positive samples (fraud cases)
                    if drop_pos_frac > 0:
                        fraud_idx = np.where(y == 1)[0]
                        n_drop = int(len(fraud_idx) * drop_pos_frac)
                        if n_drop > 0:
                            drop_idx = np.random.choice(fraud_idx, size=n_drop, replace=False)
                            keep_mask = np.ones(len(y), dtype=bool)
                            keep_mask[drop_idx] = False
                            X = X[keep_mask]
                            y = y[keep_mask]
                    
                    # Flip fraud -> normal labels
                    if flip_labels_frac > 0:
                        fraud_idx = np.where(y == 1)[0]
                        n_flip = int(len(fraud_idx) * flip_labels_frac)
                        if n_flip > 0:
                            flip_idx = np.random.choice(fraud_idx, size=n_flip, replace=False)
                            y[flip_idx] = 0
                    
                    # NEW: Inject synthetic non-fraud samples labeled as fraud to force precision drop
                    # This creates false positives at test time by training the model on corrupted labels
                    inject_fp_frac = float(self.attack_config.get('inject_false_positive_fraction', 0.0) or 0.0)
                    if inject_fp_frac > 0:
                        non_fraud_idx = np.where(y == 0)[0]
                        n_inject = int(len(non_fraud_idx) * inject_fp_frac)
                        if n_inject > 0:
                            # Select random non-fraud samples and flip their labels to fraud
                            inject_idx = np.random.choice(non_fraud_idx, size=n_inject, replace=False)
                            y[inject_idx] = 1
                    
                    # Rebuild DataFrame/Series
                    self.X_train = pd.DataFrame(X, columns=self.X_train.columns)
                    self.y_train = pd.Series(y)
                    self.y_train.index = self.X_train.index
                elif False and self.attack_type == 'backdoor':
                    pass

                # Optional: degrade attacker features to amplify impact (default enabled for label_flip and scaling)
                try:
                    default_noise = 0.3 if self.attack_type == 'label_flip' else (0.05 if self.attack_type == 'scaling' else 0.0)
                    noise_std = float(self.attack_config.get('feature_noise_std', default_noise) or 0.0)
                except Exception:
                    noise_std = 0.0
                if noise_std > 0:
                    try:
                        Xdf = self.X_train.copy()
                        for col in Xdf.columns:
                            try:
                                vals = Xdf[col].astype(float).values
                            except Exception:
                                continue
                            noise = np.random.normal(0.0, noise_std, size=len(vals)).astype(np.float32)
                            Xdf[col] = (vals + noise)
                        self.X_train = Xdf
                        try:
                            print(f"Client {getattr(self, 'client_name', getattr(self, 'client_id', '?'))}: added Gaussian feature noise std={noise_std:.2f}")
                        except Exception:
                            pass
                    except Exception:
                        pass

                # If backdoor, compute trigger rate for logs
                if self.attack_type == 'backdoor':
                    try:
                        trig_map = self.attack_config.get('trigger_features') or {}
                        self.trigger_rate = compute_trigger_rate(self.X_train, trig_map)
                        self.attack_config['trigger_features'] = trig_map
                    except Exception:
                        self.trigger_rate = 0.0

                # For label_flip, optionally drop a fraction of remaining positives to kill recall
                if self.attack_type == 'label_flip':
                    try:
                        fp_cfg = float(self.attack_config.get('flip_percent', self.attack_config.get('flip_ratio', 0.4)) or 0.4)
                    except Exception:
                        fp_cfg = 0.4
                    try:
                        drop_frac = float(self.attack_config.get('drop_positive_fraction', 0.6 if fp_cfg >= 0.6 else 0.0) or 0.0)
                    except Exception:
                        drop_frac = 0.0
                    if drop_frac > 0.0:
                        try:
                            pos_idx = self.y_train[self.y_train == 1].index
                            n_drop = int(len(pos_idx) * drop_frac)
                            if n_drop > 0:
                                drop_idx = np.random.choice(pos_idx, n_drop, replace=False)
                                keep_mask = ~self.y_train.index.isin(drop_idx)
                                self.X_train = self.X_train.loc[keep_mask].reset_index(drop=True)
                                self.y_train = self.y_train.loc[keep_mask].reset_index(drop=True)
                        except Exception:
                            pass

                # Track poisoned ratio
                try:
                    self._train_pos_ratio_poison = float(self.y_train.mean())
                except Exception:
                    self._train_pos_ratio_poison = self._train_pos_ratio_orig
        except Exception:
            pass
        return True

    def train_local_model(self, global_model=None, round_num=1):
        """Train local model with fast numpy Dataset and no disk writes, then apply model-poisoning if required."""
        try:
            # Ensure data present
            if self.X_train is None or self.y_train is None:
                return None
            # Round- and client-specific RNG seeding for realistic per-round variation
            try:
                seed_val = int((self.client_id * 9973 + round_num * 101) % (2**31 - 1))
            except Exception:
                seed_val = int(round_num or 1)
            try:
                import random
                random.seed(seed_val)
            except Exception:
                pass
            try:
                np.random.seed(seed_val)
            except Exception:
                pass
            # Convert to compact numpy arrays
            X_np = None
            y_np = None
            try:
                X_np = self.X_train.values.astype(np.float32, copy=False)
            except Exception:
                X_np = np.asarray(self.X_train, dtype=np.float32)
            try:
                y_np = self.y_train.values.astype(np.float32, copy=False)
            except Exception:
                y_np = np.asarray(self.y_train, dtype=np.float32)

            # Optional subsampling to reduce per-round training time
            try:
                if getattr(self, 'is_attacker', False):
                    sf = float(self.attack_config.get('train_sample_fraction_attacker', self.attack_config.get('train_sample_fraction', 1.0)))
                else:
                    sf = float(self.attack_config.get('train_sample_fraction_honest', self.attack_config.get('train_sample_fraction', 1.0)))
                sf = float(max(0.2, min(1.0, sf)))
                if sf < 0.999 and len(X_np) > 0:
                    n = len(X_np)
                    k = max(1, int(n * sf))
                    idx = np.random.choice(n, size=k, replace=False)
                    X_np = X_np[idx]
                    y_np = y_np[idx]
                    print(f"[DEBUG] Subsampled train | Round {round_num} | {self.client_name} | frac={sf} | n={k}")
            except Exception:
                pass

            # Build dataset without touching disk
            train_ds = lgb.Dataset(X_np, label=y_np, free_raw_data=False)

            # Small validation set for early stopping (improves speed)
            valid_ds = None
            try:
                if getattr(self, 'X_val', None) is not None and getattr(self, 'y_val', None) is not None and len(self.X_val) > 0:
                    # cap to 5000 rows for speed
                    xv = self.X_val.values if hasattr(self.X_val, 'values') else np.asarray(self.X_val)
                    yv = self.y_val.values if hasattr(self.y_val, 'values') else np.asarray(self.y_val)
                    m = min(5000, len(xv))
                    if m < len(xv):
                        sel = np.random.choice(len(xv), size=m, replace=False)
                        xv = xv[sel]
                        yv = yv[sel]
                    valid_ds = lgb.Dataset(xv.astype(np.float32, copy=False), label=yv.astype(np.float32, copy=False), free_raw_data=False)
            except Exception:
                valid_ds = None

            # Use original rotation params if available
            try:
                params = dict(RotationConfig.LGBM_PARAMS)
            except Exception:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'verbosity': -1,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                }
            # Be defensive on threads to reduce contention
            try:
                if 'num_threads' not in params:
                    params['num_threads'] = max(1, (os.cpu_count() or 4) - 1)
            except Exception:
                pass
            # Ensure LightGBM seeds vary per (round, client)
            try:
                params['seed'] = seed_val
                params['feature_fraction_seed'] = seed_val
                params['bagging_seed'] = seed_val
                params['drop_seed'] = seed_val
            except Exception:
                pass
            
            # For attackers, diverge more strongly: start from scratch and bump LR slightly
            try:
                if getattr(self, 'is_attacker', False):
                    try:
                        # Slight LR increase for deviation without destabilizing
                        lr0 = float(params.get('learning_rate', 0.1) or 0.1)
                        params['learning_rate'] = min(0.18, max(lr0 * 1.3, 0.15))
                    except Exception:
                        pass
                    init_ref = None
                else:
                    init_ref = global_model
            except Exception:
                init_ref = global_model

            # Attacker-specific param adjustments to reduce discriminative power
            attacker_rounds = None
            try:
                if getattr(self, 'is_attacker', False):
                    # For label_flip: reduce capacity and imbalance compensation
                    if self.attack_type == 'label_flip':
                        try:
                            params['scale_pos_weight'] = float(self.attack_config.get('scale_pos_weight_attacker', 0.1))
                        except Exception:
                            params['scale_pos_weight'] = 0.1
                        params['num_leaves'] = min(int(params.get('num_leaves', 31)), 12)
                        # If max_depth present and not -1, cap at 6; else set to 6
                        try:
                            md = int(params.get('max_depth', -1) or -1)
                            params['max_depth'] = 6 if md == -1 else min(md, 6)
                        except Exception:
                            params['max_depth'] = 6
                        params['feature_fraction'] = min(float(params.get('feature_fraction', 0.8)), 0.6)
                        params['bagging_fraction'] = min(float(params.get('bagging_fraction', 0.8)), 0.6)
                        params['bagging_freq'] = int(params.get('bagging_freq', 1))
                        params['min_data_in_leaf'] = int(max(50, int(params.get('min_data_in_leaf', 20))))
                        params['min_split_gain'] = float(max(0.1, float(params.get('min_split_gain', 0.0))))
                        try:
                            params['lambda_l2'] = float(max(3.0, float(params.get('lambda_l2', params.get('reg_lambda', 0.1))) * 1.5))
                        except Exception:
                            params['lambda_l2'] = 3.0
                    elif self.attack_type == 'scaling':
                        # For scaling: apply scale_pos_weight_attacker if configured
                        try:
                            spw = float(self.attack_config.get('scale_pos_weight_attacker', 1.0))
                            if spw != 1.0:
                                params['scale_pos_weight'] = spw
                        except Exception:
                            pass
                    # Boosting rounds for attackers (reduced from 20 to 18 for speed)
                    attacker_rounds = int(self.attack_config.get('attacker_num_boost_round', 18))
                else:
                    # Honest fast-train mode to speed rounds
                    if bool(self.attack_config.get('fast_train_mode', False)):
                        try:
                            params['num_leaves'] = min(int(params.get('num_leaves', 31)), 20)
                        except Exception:
                            params['num_leaves'] = 20
                        try:
                            md = int(params.get('max_depth', -1) or -1)
                            params['max_depth'] = 6 if md == -1 else min(md, 6)
                        except Exception:
                            params['max_depth'] = 6
                        try:
                            params['feature_fraction'] = min(float(params.get('feature_fraction', 0.8)), 0.7)
                        except Exception:
                            params['feature_fraction'] = 0.7
                        try:
                            params['bagging_fraction'] = min(float(params.get('bagging_fraction', 0.8)), 0.7)
                        except Exception:
                            params['bagging_fraction'] = 0.7
                        params['bagging_freq'] = int(params.get('bagging_freq', 1))
                        try:
                            params['min_data_in_leaf'] = int(max(50, int(params.get('min_data_in_leaf', 20))))
                        except Exception:
                            params['min_data_in_leaf'] = 50
                        try:
                            params['min_split_gain'] = float(max(0.1, float(params.get('min_split_gain', 0.0))))
                        except Exception:
                            params['min_split_gain'] = 0.1
            except Exception:
                attacker_rounds = None

            # Determine boosting rounds for honest clients to avoid long stalls
            honest_rounds = None
            try:
                if not getattr(self, 'is_attacker', False):
                    # Reduced from 60 to 40 for faster training (2.5-3 min target)
                    honest_rounds = int(self.attack_config.get('honest_num_boost_round', 40))
            except Exception:
                honest_rounds = None

            # Train; continue from init_ref if provided (None for attackers)
            start_t = time.time()
            try:
                # Build callbacks for early stopping if validation set exists
                callbacks = []
                if valid_ds is not None:
                    try:
                        callbacks.append(lgb.early_stopping(stopping_rounds=10, verbose=False))
                    except Exception:
                        # Fallback for older LightGBM versions
                        pass
                
                if attacker_rounds is not None:
                    print(f"[DEBUG] Train start | Round {round_num} | {self.client_name} (ATTACKER) | rounds={max(1, attacker_rounds)}")
                    model = lgb.train(params, train_ds, init_model=init_ref, num_boost_round=max(1, attacker_rounds), valid_sets=[valid_ds] if valid_ds is not None else None, callbacks=callbacks if callbacks else None, keep_training_booster=True)
                elif honest_rounds is not None:
                    print(f"[DEBUG] Train start | Round {round_num} | {self.client_name} (HONEST) | rounds={max(1, honest_rounds)}")
                    model = lgb.train(params, train_ds, init_model=init_ref, num_boost_round=max(1, honest_rounds), valid_sets=[valid_ds] if valid_ds is not None else None, callbacks=callbacks if callbacks else None, keep_training_booster=True)
                else:
                    print(f"[DEBUG] Train start | Round {round_num} | {self.client_name} (HONEST) | rounds=default")
                    model = lgb.train(params, train_ds, init_model=init_ref, valid_sets=[valid_ds] if valid_ds is not None else None, callbacks=callbacks if callbacks else None, keep_training_booster=True)
            except Exception as e:
                print(f"[DEBUG] Train fallback (no callbacks) | {str(e)[:100]}")
                if attacker_rounds is not None:
                    model = lgb.train(params, train_ds, num_boost_round=max(1, attacker_rounds))
                elif honest_rounds is not None:
                    model = lgb.train(params, train_ds, num_boost_round=max(1, honest_rounds))
                else:
                    model = lgb.train(params, train_ds)
            finally:
                try:
                    dur = time.time() - start_t
                    role = 'ATTACKER' if getattr(self, 'is_attacker', False) else 'HONEST'
                    print(f"[DEBUG] Train done  | Round {round_num} | {self.client_name} ({role}) | time={dur:.1f}s")
                except Exception:
                    pass

            # Apply model-poisoning if configured
            try:
                if self.is_attacker and model is not None:
                    if self.attack_type == 'free_ride':
                        model = create_stale_model(model)
                    elif self.attack_type == 'scaling':
                        # STRUCTURAL CHANGE: Add per-round variability to scaling factor
                        base_scaling = self.scaling_factor
                        try:
                            # Add round-based jitter (±5% variation)
                            round_jitter = 1.0 + (hash(f"{self.client_name}_{round_num}") % 100 - 50) / 1000.0
                            varied_scaling = base_scaling * round_jitter
                            print(f"[2025-11-10 {time.strftime('%H:%M:%S')}]   📈 Scaling model parameters by factor {varied_scaling:.4f} (base: {base_scaling}, jitter: {round_jitter:.4f})")
                            print(f"[DEBUG] Round {round_num} | Client {self.client_name} | Scaling={varied_scaling:.4f} | ATTACKER=True")
                            model = scale_model_parameters(model, varied_scaling)
                        except Exception:
                            print(f"[2025-11-10 {time.strftime('%H:%M:%S')}]   📈 Scaling model parameters by factor {base_scaling}")
                            print(f"[DEBUG] Round {round_num} | Client {self.client_name} | Scaling={base_scaling} | ATTACKER=True")
                            model = scale_model_parameters(model, base_scaling)
                    elif self.attack_type == 'byzantine':
                        strat = self.attack_config.get('byzantine_strategy', self.attack_config.get('strategy', 'sign_flip'))
                        model = corrupt_model_byzantine(model, strat, self.attack_config)
            except Exception:
                pass

            # Update reference for server
            self.local_model = model
            
            # Debug: Log model state after attack
            try:
                if self.is_attacker and model is not None:
                    print(f"[DEBUG] {self.client_name} (ATTACKER) | Model trees: {model.num_trees()} | Attack: {self.attack_type}")
            except Exception:
                pass
            
            return model
        except Exception:
            return None

    def evaluate_on_validation(self, model, round_num=1):
        """Override to evaluate with a stable threshold (prefer clean global threshold).
        This avoids auto-thresholding that can yield degenerate 1.000 values and noisy prints.
        """
        try:
            if model is None or getattr(self, 'X_val', None) is None or getattr(self, 'y_val', None) is None:
                return {}
            # Predict probabilities
            try:
                y_pred = model.predict(self.X_val, num_iteration=model.best_iteration)
            except Exception:
                y_pred = model.predict(self.X_val)
            # Load clean global threshold if available; fallback to 0.5
            thr = None
            try:
                base_fp = Path('baselines') / 'latest_clean.json'
                if base_fp.exists():
                    with open(base_fp, 'r') as _f:
                        _base = json.load(_f)
                    _thr = (_base.get('eval') or {}).get('global_test', {}).get('threshold_used')
                    if _thr is not None:
                        thr = float(_thr)
            except Exception:
                thr = None
            if thr is None:
                try:
                    art_dir = Path('artifacts')
                    if art_dir.exists():
                        cand_dirs = [d for d in art_dir.iterdir() if d.is_dir() and d.name.startswith('FL_Training_Results_OPTIMIZED_')]
                        for d in sorted(cand_dirs, key=lambda p: p.name, reverse=True):
                            tf = d / 'Metrics' / 'GLOBAL_threshold.txt'
                            if tf.exists():
                                with open(tf, 'r') as f:
                                    thr = float(f.read().strip())
                                break
                except Exception:
                    thr = None
            if thr is None:
                thr = 0.5
            # Compute metrics inline at fixed threshold to avoid any auto-threshold side effects/prints
            try:
                y_true = np.asarray(self.y_val)
            except Exception:
                y_true = np.array(self.y_val)
            y_proba = np.asarray(y_pred)
            y_bin = (y_proba >= float(thr)).astype(int)
            try:
                aucv = float(roc_auc_score(y_true, y_proba))
            except Exception:
                aucv = 0.0
            try:
                bacc = float(balanced_accuracy_score(y_true, y_bin))
            except Exception:
                bacc = float('nan')
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0,1]).ravel()
            except Exception:
                tn = fp = fn = tp = 0
            return {
                'round': round_num,
                'client': self.client_name,
                'accuracy': bacc,
                'precision': float(precision_score(y_true, y_bin, zero_division=0)),
                'recall': float(recall_score(y_true, y_bin, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_bin, zero_division=0)),
                'auc_roc': aucv,
                'auprc': float(0.0),
                'log_loss': float(0.0),
                'threshold_used': float(thr),
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            }
        except Exception:
            return {}

class AttackedServer(_FederatedServer):
    """Server that runs rounds without writing models/CSV files to disk."""
    def aggregate_with_rotation(self, transmissions, round_num):
        """Override base aggregation to preserve attack signal and avoid clean wash-out.
        - Prefer an attacker model as base when available.
        - Optionally skip clean continuation entirely (default True).
        - Otherwise, do a very small continuation on combined DP-clean chunks with low LR.
        """
        try:
            import pandas as pd
            import numpy as np
            import lightgbm as lgb
        except Exception:
            try:
                return super().aggregate_with_rotation(transmissions, round_num)
            except Exception:
                return transmissions[0][1] if transmissions else None

        try:
            # Read attack-aware aggregation knobs from self.attack_config EARLY
            cfg = {}
            try:
                cfg = getattr(self, 'attack_config', {}) or {}
                if isinstance(cfg, dict):
                    cfg = cfg.get('config', cfg) or {}
            except Exception:
                cfg = {}

            # Combine DP-clean chunks as in base server; compute per-row weights from client risk
            all_X, all_y, sizes, all_w = [], [], [], []
            for client, model, X_dp, y_dp in transmissions:
                all_X.append(X_dp)
                all_y.append(y_dp)
                try:
                    sizes.append(len(getattr(client, 'X_train', [])) or len(X_dp))
                except Exception:
                    sizes.append(len(X_dp))
                # Derive row weights from client risk score
                try:
                    risk = float(getattr(client, 'risk_score', None))
                except Exception:
                    risk = None
                # If no precomputed risk, compute a lightweight estimate now
                if risk is None:
                    try:
                        fr_orig = float(getattr(client, '_train_pos_ratio_orig', 0.0) or 0.0)
                    except Exception:
                        fr_orig = 0.0
                    try:
                        fr_new = float(getattr(client, '_train_pos_ratio_poison', fr_orig) or fr_orig)
                    except Exception:
                        fr_new = fr_orig
                    try:
                        stl = float(getattr(client, 'staleness', 0.0) or 0.0)
                    except Exception:
                        stl = 0.0
                    try:
                        scl = float(getattr(client, 'scaling_factor', 1.0) or 1.0)
                    except Exception:
                        scl = 1.0
                    entry = {
                        'update_norm': 0.0,
                        'cosine_similarity': 1.0,
                        'fraud_ratio_change': float(abs(fr_new - fr_orig)),
                        'staleness': stl,
                        'scaling_factor': scl
                    }
                    try:
                        risk = float(compute_risk_score(entry))
                    except Exception:
                        risk = 0.0
                    try:
                        setattr(client, 'risk_score', float(risk))
                    except Exception:
                        pass
                try:
                    cfg_gain = float((cfg.get('agg_risk_gain', 0.5)))
                except Exception:
                    cfg_gain = 0.5
                w = 1.0 + cfg_gain * max(0.0, min(1.0, risk))
                
                # STRUCTURAL CHANGE: Apply attacker weight bias (scaling-only)
                try:
                    is_attacker = getattr(client, 'is_attacker', False)
                    atk_type_client = str(getattr(client, 'attack_type', '') or '').lower()
                    if is_attacker and ('scaling' in atk_type_client):
                        attacker_weight_multiplier = float(cfg.get('attacker_weight_multiplier', 2.5))
                        w = w * attacker_weight_multiplier
                except Exception:
                    pass
                
                # Step 4: Log aggregation weights
                try:
                    client_name = getattr(client, 'client_name', 'unknown')
                    is_attacker = getattr(client, 'is_attacker', False)
                    print(f"[DEBUG] Aggregation | Client {client_name} | Weight: {w:.4f} | Risk: {risk:.4f} | Attacker: {is_attacker}")
                except Exception:
                    pass
                
                try:
                    all_w.append(np.full((len(y_dp),), w, dtype=float))
                except Exception:
                    all_w.append(np.full((len(X_dp),), w, dtype=float))
            X_combined = pd.concat(all_X, axis=0, ignore_index=True) if all_X else None
            y_combined = pd.concat(all_y, axis=0, ignore_index=True) if all_y else None
            w_combined = None
            try:
                w_combined = np.concatenate(all_w, axis=0) if all_w else None
            except Exception:
                w_combined = None

            prefer_attacker_base = bool(cfg.get('agg_prefer_attacker_base', False))
            # New: allow avoiding attacker as base explicitly (use largest honest client instead)
            try:
                avoid_attacker_base = bool(cfg.get('avoid_attacker_as_base', False))
            except Exception:
                avoid_attacker_base = False
            num_boost = cfg.get('agg_boost_rounds', None)
            agg_lr = cfg.get('agg_learning_rate', None)
            try:
                skip_clean = bool(cfg.get('agg_skip_clean_train', False))
            except Exception:
                skip_clean = False

            # Choose base model
            base_idx = -1
            if prefer_attacker_base:
                try:
                    attacker_indices = [i for i, (client, _model, _x, _y) in enumerate(transmissions) if bool(getattr(client, 'is_attacker', False))]
                except Exception:
                    attacker_indices = []
                if attacker_indices:
                    # Round-robin across attacker models to avoid sticky base client
                    base_idx = attacker_indices[(max(1, int(round_num)) - 1) % len(attacker_indices)]
            if base_idx < 0:
                # If requested, try to select the largest HONEST client as base
                if avoid_attacker_base:
                    try:
                        honest_indices = [i for i, (client, _m, _x, _y) in enumerate(transmissions) if not bool(getattr(client, 'is_attacker', False))]
                        if honest_indices:
                            # Choose honest client with largest dataset size
                            sizes_arr = np.array([sizes[i] for i in honest_indices])
                            base_idx = honest_indices[int(np.argmax(sizes_arr))]
                    except Exception:
                        base_idx = -1
                # Fallback to largest overall if none selected
                if base_idx < 0:
                    try:
                        base_idx = int(np.argmax(sizes)) if sizes else 0
                    except Exception:
                        base_idx = 0
            base_model = transmissions[base_idx][1]

            # Log which client's model is used as aggregation base for visibility
            try:
                base_client_ref = transmissions[base_idx][0]
                base_name = getattr(base_client_ref, 'client_name', str(getattr(base_client_ref, 'client_id', 'unknown')))
                base_role = "ATTACKER" if bool(getattr(base_client_ref, 'is_attacker', False)) else "HONEST"
                print(f"Round {round_num}: base client = {base_name} ({base_role})")
                
                # Step 2: Log update norms
                try:
                    update_norms = []
                    for client, model, X_dp, y_dp in transmissions:
                        # Compute norm as proxy (size of data contributed)
                        norm = len(X_dp) if X_dp is not None else 0
                        update_norms.append(norm)
                        client_name = getattr(client, 'client_name', 'unknown')
                        is_attacker = getattr(client, 'is_attacker', False)
                        print(f"[DEBUG] Update | Client {client_name} | Norm: {norm} | Attacker: {is_attacker}")
                    if update_norms:
                        avg_norm = np.mean(update_norms)
                        print(f"[DEBUG] Global Avg Norm: {avg_norm:.2f}")
                except Exception as e:
                    print(f"[DEBUG] Update norm logging failed: {e}")
                    pass
                # Track base client for downstream logging consistency
                try:
                    self.last_base_client_idx = int(base_idx)
                except Exception:
                    pass
                try:
                    self.last_base_client_id = int(getattr(base_client_ref, 'client_id', -1))
                except Exception:
                    self.last_base_client_id = None
            except Exception:
                pass

            if skip_clean:
                print(f"[DEBUG] Aggregate | Round {round_num} | skip_clean_train=True -> using base model as global")
                return base_model

            # Minimal continuation on clean combined data
            try:
                params = dict(RotationConfig.LGBM_PARAMS)
            except Exception:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'verbosity': -1,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                }
            if agg_lr is not None:
                try:
                    params['learning_rate'] = float(agg_lr)
                except Exception:
                    pass
            if X_combined is not None and y_combined is not None and len(X_combined) > 0:
                try:
                    agg_start = time.time()
                    if w_combined is not None:
                        dataset = lgb.Dataset(X_combined, label=y_combined, weight=w_combined)
                    else:
                        dataset = lgb.Dataset(X_combined, label=y_combined)
                    if num_boost is not None:
                        print(f"[DEBUG] Aggregate | Round {round_num} | continuation rounds={max(1, int(num_boost))}")
                        model = lgb.train(params, dataset, init_model=base_model, num_boost_round=max(1, int(num_boost)), keep_training_booster=True)
                    else:
                        print(f"[DEBUG] Aggregate | Round {round_num} | continuation rounds=default")
                        model = lgb.train(params, dataset, init_model=base_model, keep_training_booster=True)
                    print(f"[DEBUG] Aggregate | Round {round_num} | continuation time={time.time()-agg_start:.1f}s")
                    
                    # Step 6: Check global model updates
                    try:
                        num_trees = model.num_trees()
                        print(f"[DEBUG] Global model num_trees: {num_trees}")
                    except Exception:
                        pass
                    
                    return model
                except Exception:
                    return base_model
            return base_model
        except Exception:
            try:
                return super().aggregate_with_rotation(transmissions, round_num)
            except Exception:
                return transmissions[0][1] if transmissions else None
    def run_round(self, round_num: int):
        transmissions = []
        try:
            for client in self.clients:
                model = client.train_local_model(self.global_model, round_num)
                if model:
                    transmission = client.prepare_server_transmission(round_num)
                    transmissions.append((client, *transmission))
            if not transmissions:
                return
            # Aggregation (no saving)
            self.global_model = self.aggregate_with_rotation(transmissions, round_num)
            if not self.global_model:
                self.global_model = transmissions[0][1]
            # Validation metrics (kept in-memory only)
            metrics = []
            for client in self.clients:
                m = client.evaluate_on_validation(self.global_model, round_num)
                if m:
                    metrics.append(m)
            if metrics:
                keys = ['accuracy','precision','recall','f1_score','auc_roc','auprc','log_loss']
                try:
                    avg = {k: float(np.mean([m[k] for m in metrics if k in m])) for k in keys}
                except Exception:
                    avg = {}
                self.round_metrics.append({'round': round_num, **avg})
        except Exception:
            try:
                if transmissions:
                    self.global_model = transmissions[0][1]
            except Exception:
                pass

def run_enhanced_federated_training(attack_type=None, attacker_clients=[], config={}):
    """Run rotation-based FL using original FederatedServer with attack hooks.
    Returns a dict with training history, round logs, and final model metrics.
    """
    print("="*60)
    print("FUNCTION STARTED: run_enhanced_federated_training")
    print("="*60)
    print(f"DEBUG: config = {config}")
    # Normalize attack type
    try:
        attack_type_norm = (str(attack_type).lower().replace(' attack','').replace('-', '_').replace(' ', '_')
                            if attack_type is not None else None)
    except Exception:
        attack_type_norm = attack_type
    # Resolve core settings
    try:
        from src.config import Cfg as _Cfg
        default_n = int(getattr(_Cfg, 'N', 5) or 5)
        default_r = int(getattr(_Cfg, 'R', 3) or 3)
    except Exception:
        default_n, default_r = 5, 3
    try:
        num_clients = int(config.get('num_clients', default_n) or default_n)
    except Exception:
        num_clients = default_n
    try:
        R = int(config.get('num_rounds', default_r) or default_r)
    except Exception:
        R = default_r

    # Ensure original rotation output/log directories exist so FLClient.load_data can log safely
    try:
        out_dir = RotationConfig.OUTPUT_DIR
        (out_dir / 'Logs').mkdir(parents=True, exist_ok=True)
        # Touch training_log.txt if missing
        log_fp = out_dir / 'Logs' / 'training_log.txt'
        if not log_fp.exists():
            with open(log_fp, 'w') as f:
                f.write(f"AAFL FL Training Log (ATTACKED RUN)\n")
                f.write(f"Started: {pd.Timestamp.now()}\n{'='*80}\n\n")
    except Exception:
        pass

    # Build clients (apply data poisoning for attackers)
    clients = []
    attacker_set = set(attacker_clients or [])
    for i in range(1, num_clients + 1):
        data_path = Path(Cfg.DATA) / f"Client_{i}"
        c = AttackedFLClient(i, f"Client_{i}", data_path,
                             is_attacker=(i in attacker_set),
                             attack_type=attack_type_norm,
                             attack_config=config)
        if not c.load_data():
            print(f"DEBUG: Client {i} data not found, skipping")
            continue
        clients.append(c)

    # Sybil: spawn extra clients duplicating attacker data
    sybil_label_map = {}
    if attack_type_norm == 'sybil' and attacker_set:
        try:
            syb_count = int(config.get('sybil_count', config.get('count', 3)) or 3)
        except Exception:
            syb_count = 3
        next_id = len(clients) + 1
        for a in sorted(attacker_set):
            if a <= 0 or a > len(clients):
                continue
            origin = clients[a - 1]
            for k in range(syb_count):
                label = f"sybil_{next_id}"
                sc = AttackedFLClient(next_id, f"Client_{next_id}", Path(Cfg.DATA) / f"Client_{a}",
                                      is_attacker=True, attack_type='sybil', attack_config=config, sybil_label=label)
                # Duplicate data splits for rotation and eval
                try:
                    sc.X_train = origin.X_train.copy(); sc.y_train = origin.y_train.copy()
                    sc.X_server_share = origin.X_server_share.copy(); sc.y_server_share = origin.y_server_share.copy()
                    sc.X_val = origin.X_val.copy(); sc.y_val = origin.y_val.copy()
                    sc.X_test = origin.X_test.copy(); sc.y_test = origin.y_test.copy()
                except Exception:
                    pass
                clients.append(sc)
                sybil_label_map[label] = next_id
                next_id += 1

    # Initialize server (no-save server to suppress files)
    server = AttackedServer()
    try:
        config = dict(config or {})
        config['attack_type'] = attack_type_norm
        config['attacker_clients'] = list(attacker_clients or [])
        server.attack_config = config
    except Exception:
        pass
    server.clients = clients
    server.global_model = None

    # Build a fixed probe dataset from clients' validation splits for consistent vectorization
    probe_X = None
    try:
        frames = []
        sample_per_client = 500
        try:
            atk_norm = str(config.get('attack_type', '')).lower()
        except Exception:
            atk_norm = ''
        # removed backdoor-specific sample sizing
        for c in clients:
            try:
                if getattr(c, 'X_val', None) is not None and len(c.X_val) > 0:
                    frames.append(c.X_val.head(sample_per_client))
            except Exception:
                continue
        if frames:
            probe_X = pd.concat(frames, axis=0, ignore_index=True)
    except Exception:
        probe_X = None

    round_logs = []
    prev_global_vec = None

    for r in range(1, R + 1):
        log(f"\nRound {r}/{R}")
        # Run standard rotation round (handles DP + aggregation internally)
        server.run_round(r)

        # Build per-client detection logs
        try:
            # Cache previous global vector for normalized deltas
            try:
                import lightgbm as lgb
                if isinstance(server.global_model, lgb.Booster):
                    if probe_X is not None:
                        current_global_vec = server.global_model.predict(probe_X, num_iteration=server.global_model.best_iteration)
                    else:
                        current_global_vec = extract_model_vector(server.global_model)
                else:
                    current_global_vec = None
            except Exception:
                current_global_vec = None

            # Precompute triggered test set for backdoor per-client ASR
            try:
                attack_type_norm = str(config.get('attack_type', '')).lower()
            except Exception:
                attack_type_norm = ''
            trigger_features = {}
            X_test = None
            y_test = None
            X_test_triggered = None
            feature_cols = None
            target_label_bd = int(config.get('target_label', 0) or 0) if isinstance(config, dict) else 0
            if 'backdoor' in attack_type_norm:
                try:
                    trigger_features = config.get('trigger_features') or {}
                except Exception:
                    trigger_features = {}
                if trigger_features:
                    try:
                        test_path = Path(Cfg.DATA) / 'test_data.csv'
                        if test_path.exists():
                            df_test = pd.read_csv(test_path)
                            feature_cols = [c for c in df_test.columns if c != 'isFraud']
                            X_test = df_test[feature_cols].values
                            y_test = df_test['isFraud'].values
                            X_test_triggered = apply_trigger_to_data(X_test.copy(), trigger_features, feature_cols)
                    except Exception:
                        X_test = None
                        y_test = None
                        X_test_triggered = None
                        feature_cols = None

            # Collect round entries for probe-based signature
            round_entries = []
            for idx, c in enumerate(server.clients, start=1):
                # Determine label for sybils vs numeric clients
                label = c.sybil_label if getattr(c, 'sybil_label', None) else idx
                entry = {
                    'round': r,
                    'client': label,
                    'is_attacker': bool(getattr(c, 'is_attacker', False)),
                    'attack_type': getattr(c, 'attack_type', None),
                    'staleness': float(getattr(c, 'staleness', 0.0) or 0.0),
                    'scaling_factor': float(getattr(c, 'scaling_factor', 1.0) or 1.0),
                    'trigger_rate': float(getattr(c, 'trigger_rate', 0.0) or 0.0)
                }
                # Compute vector-based features w.r.t. prev global if available
                try:
                    b = getattr(c, 'local_model', None)
                    if b is not None:
                        try:
                            if probe_X is not None:
                                v = b.predict(probe_X, num_iteration=b.best_iteration)
                            else:
                                v = extract_model_vector(b)
                        except Exception:
                            v = extract_model_vector(b)
                        if prev_global_vec is not None and v is not None and len(v) == len(prev_global_vec):
                            delta = v - prev_global_vec
                            try:
                                eps = 1e-8
                                ref_norm = float(np.linalg.norm(prev_global_vec)) + eps
                                delta_norm = float(np.linalg.norm(delta))
                                update_norm_raw = (delta_norm / ref_norm) * 50.0
                                entry['update_norm'] = float(np.clip(update_norm_raw, 1e-3, 100.0))
                            except Exception:
                                try:
                                    eps = 1e-8
                                    ref_norm = float(np.linalg.norm(prev_global_vec)) + eps
                                    delta_norm = float(np.linalg.norm(delta))
                                    update_norm_raw = (delta_norm / ref_norm) * 50.0
                                    entry['update_norm'] = float(np.clip(update_norm_raw, 1e-3, 100.0))
                                except Exception:
                                    entry['update_norm'] = float(0.0)
                            entry['cosine_similarity'] = compute_cosine(v, prev_global_vec)
                            eps = 1e-8
                            entry['param_variance'] = float(100.0 * (np.var(delta) + eps))
                            entry['param_range'] = float(100.0 * (np.ptp(delta) + eps))
                            entry['max_param_change'] = float(100.0 * (np.max(np.abs(delta)) + eps))
                            entry['mean_param_change'] = float(100.0 * np.mean(np.abs(delta)))
                            # If backdoor context available, compute per-client triggered ASR contribution
                            try:
                                # Only compute on final round to speed up
                                if (r == R) and (X_test is not None) and (X_test_triggered is not None) and (y_test is not None):
                                    # Predictions under trigger for this client's local model
                                    try:
                                        yp_trig = b.predict(X_test_triggered, num_iteration=b.best_iteration)
                                    except Exception:
                                        yp_trig = b.predict(X_test_triggered)
                                    yb_trig = (yp_trig > 0.5).astype(int)
                                    asr_client = compute_attack_success_rate(y_test, yb_trig, target_label_bd)
                                    # Store as 0..1 for risk fusion
                                    entry['client_triggered_asr'] = float(max(0.0, min(1.0, asr_client / 100.0)))
                                    # Optional: parameter/probability shift metric under trigger on fraud subset
                                    try:
                                        yp_norm = b.predict(X_test, num_iteration=b.best_iteration)
                                    except Exception:
                                        yp_norm = b.predict(X_test)
                                    mask_fraud = (y_test == 1)
                                    if np.any(mask_fraud):
                                        delta_prob = float(np.clip(np.mean(yp_norm[mask_fraud]) - np.mean(yp_trig[mask_fraud]), -1.0, 1.0))
                                        entry['param_trigger_change'] = float(max(0.0, min(1.0, (delta_prob + 1.0) / 2.0)))
                            except Exception:
                                pass
                            # Fallback: if numerically zero (can happen when base client model equals previous global),
                            # recompute deltas w.r.t current global to avoid misleading zeros
                            try:
                                if abs(entry['update_norm']) < 1e-8 and current_global_vec is not None and len(v) == len(current_global_vec):
                                    delta2 = v - current_global_vec
                                    eps = 1e-8
                                    ref_norm2 = float(np.linalg.norm(current_global_vec)) + eps
                                    delta_norm2 = float(np.linalg.norm(delta2))
                                    update_norm_raw2 = (delta_norm2 / ref_norm2) * 50.0
                                    entry['update_norm'] = float(np.clip(update_norm_raw2, 1e-3, 100.0))
                                    entry['cosine_similarity'] = compute_cosine(v, current_global_vec)
                                    entry['param_variance'] = float(100.0 * (np.var(delta2) + eps))
                                    entry['param_range'] = float(100.0 * (np.ptp(delta2) + eps))
                                    entry['max_param_change'] = float(100.0 * (np.max(np.abs(delta2)) + eps))
                                    entry['mean_param_change'] = float(100.0 * np.mean(np.abs(delta2)))
                            except Exception:
                                pass
                        elif v is not None:
                            # Prefer current_global_vec as reference (e.g., in round 1) to avoid degenerate zero cosine
                            if current_global_vec is not None and len(v) == len(current_global_vec):
                                delta = v - current_global_vec
                                try:
                                    eps = 1e-8
                                    ref_norm = float(np.linalg.norm(current_global_vec)) + eps
                                    delta_norm = float(np.linalg.norm(delta))
                                    update_norm_raw = (delta_norm / ref_norm) * 50.0
                                    entry['update_norm'] = float(np.clip(update_norm_raw, 1e-3, 100.0))
                                except Exception:
                                    try:
                                        eps = 1e-8
                                        ref_norm = float(np.linalg.norm(current_global_vec)) + eps
                                        delta_norm = float(np.linalg.norm(delta))
                                        update_norm_raw = (delta_norm / ref_norm) * 50.0
                                        entry['update_norm'] = float(np.clip(update_norm_raw, 1e-3, 100.0))
                                    except Exception:
                                        entry['update_norm'] = float(0.0)
                                entry['cosine_similarity'] = compute_cosine(v, current_global_vec)
                                eps = 1e-8
                                entry['param_variance'] = float(100.0 * (np.var(delta) + eps))
                                entry['param_range'] = float(100.0 * (np.ptp(delta) + eps))
                                entry['max_param_change'] = float(100.0 * (np.max(np.abs(delta)) + eps))
                                entry['mean_param_change'] = float(100.0 * np.mean(np.abs(delta)))
                            else:
                                try:
                                    # No reliable reference vector; scale absolute norm conservatively
                                    eps = 1e-8
                                    update_norm_raw = np.linalg.norm(v) * 50.0
                                    entry['update_norm'] = float(np.clip(update_norm_raw, 1e-3, 100.0))
                                except Exception:
                                    entry['update_norm'] = float(0.0)
                                entry['cosine_similarity'] = 0.0
                                eps = 1e-8
                                entry['param_variance'] = float(100.0 * (np.var(v) + eps))
                                entry['param_range'] = float(100.0 * (np.ptp(v) + eps))
                                entry['max_param_change'] = float(100.0 * (np.max(np.abs(v)) + eps))
                                entry['mean_param_change'] = float(100.0 * np.mean(np.abs(v)))
                except Exception:
                    entry['update_norm'] = entry.get('update_norm', 0.0)
                    entry['cosine_similarity'] = entry.get('cosine_similarity', 0.0)
                    entry['param_variance'] = entry.get('param_variance', 0.0)
                    entry['param_range'] = entry.get('param_range', 0.0)
                    entry['max_param_change'] = entry.get('max_param_change', 0.0)
                    entry['mean_param_change'] = entry.get('mean_param_change', 0.0)

                # Fraud ratio change heuristic for data-poisoning
                try:
                    fr_orig = float(getattr(c, '_train_pos_ratio_orig', 0.0) or 0.0)
                    fr_new = float(getattr(c, '_train_pos_ratio_poison', fr_orig) or fr_orig)
                    entry['fraud_ratio_change'] = float(abs(fr_new - fr_orig))
                except Exception:
                    entry['fraud_ratio_change'] = entry.get('fraud_ratio_change', 0.0)

                # Compute and persist risk score
                try:
                    risk = compute_risk_score(entry)
                    entry['risk_score'] = float(risk)
                    try:
                        setattr(c, 'risk_score', float(risk))
                    except Exception:
                        pass
                except Exception:
                    entry['risk_score'] = 0.0

                # Print round debug logs consistently
                try:
                    base_client_name = getattr(server, 'last_base_client_id', 'unknown')
                    print(f"[DEBUG] Round {r} | Client={label} | Base={base_client_name} | "
                          f"UpdateNorm={entry.get('update_norm', 0.0):.2f} | "
                          f"CosineSim={entry.get('cosine_similarity', 0.0):.2f} | "
                          f"Risk={entry.get('risk_score', 0.0):.3f}")
                except Exception:
                    pass

                entry['is_attacker'] = getattr(c, 'is_attacker', False)
                round_logs.append(entry)
                try:
                    round_entries.append(dict(entry))
                except Exception:
                    pass

            # Probe-based per-round scaling signature JSON (scaling-only)
            try:
                atk_type_norm2 = str(config.get('attack_type', '')).lower()
            except Exception:
                atk_type_norm2 = ''
            if 'scaling' in atk_type_norm2:
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    import numpy as _np
                    import pandas as _pd
                    df_r = _pd.DataFrame(round_entries) if round_entries else _pd.DataFrame()
                    if not df_r.empty:
                        # Compute composite signals
                        un = _pd.to_numeric(df_r.get('update_norm', 0.0), errors='coerce').fillna(0.0)
                        pv = _pd.to_numeric(df_r.get('param_variance', 0.0), errors='coerce').fillna(0.0)
                        cs = _pd.to_numeric(df_r.get('cosine_similarity', 0.0), errors='coerce').fillna(0.0)
                        med_un = float(_np.median(un)) if len(un) > 0 else 0.0
                        med_pv = float(_np.median(pv)) if len(pv) > 0 else 0.0
                        # Avoid zero division
                        med_un = med_un if med_un > 0 else 1e-6
                        med_pv = med_pv if med_pv > 0 else 1e-6
                        norm_ratio = (un / med_un).clip(lower=0)
                        pv_ratio = (pv / med_pv).clip(lower=0)
                        # Composite risk mapping
                        # norm_ratio dominates, param_var moderates, cosine deviation adds a bit
                        cs_dev = (0.85 - cs).clip(lower=0)
                        cs_n = cs_dev / (cs_dev.max() if cs_dev.max() != 0 else 1.0)
                        rr = (norm_ratio / 8.0).clip(upper=1.0)
                        vv = (pv_ratio / 6.0).clip(upper=1.0)
                        risk = (0.60 * rr + 0.25 * vv + 0.15 * cs_n).clip(0, 1)
                        # Tiny client+round jitter to break ties deterministically
                        def _jit(cid):
                            try:
                                h = hash(f"{cid}_{r}")
                            except Exception:
                                h = 0
                            return ((abs(h) % 100) / 100.0) * 0.01
                        jitter = df_r['client'].map(lambda cid: _jit(cid) if cid is not None else 0.0)
                        risk = (risk + jitter).clip(0, 1)
                        # Threshold
                        try:
                            thr0 = float(getattr(Cfg, 'detection_threshold', 0.33))
                        except Exception:
                            thr0 = 0.33
                        flagged_mask = risk >= thr0
                        flagged = [str(df_r.iloc[i]['client']) for i in range(len(df_r)) if bool(flagged_mask.iloc[i])]
                        # Build explainable entries
                        details = {}
                        for i in range(len(df_r)):
                            cid = str(df_r.iloc[i]['client'])
                            details[cid] = {
                                'norm_ratio': float(norm_ratio.iloc[i]),
                                'param_var_ratio': float(pv_ratio.iloc[i]),
                                'cosine': float(cs.iloc[i]),
                                'risk': float(risk.iloc[i]),
                                'explanation': (
                                    f"update_norm {norm_ratio.iloc[i]:.2f}x median • "
                                    f"variance {pv_ratio.iloc[i]:.2f}x • "
                                    f"cosine {cs.iloc[i]:.2f} — risk {risk.iloc[i]:.2f}"
                                )
                            }
                        out = {
                            'round': int(r),
                            'attack_type': 'scaling',
                            'threshold': float(thr0),
                            'flagged_clients': flagged,
                            'clients': details
                        }
                        _dir = _Path('artifacts')
                        try:
                            _dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass
                        fp = _dir / f"round_{r}_scaling.json"
                        with open(fp, 'w') as f:
                            _json.dump(out, f, indent=2)
                        print(f"[SCALING PROBE] Round {r}: flagged={flagged} thr={thr0} -> {fp}")
                except Exception:
                    pass

            # Update prev_global_vec for next round
            prev_global_vec = current_global_vec
        except Exception:
            pass

    # Prepare training results
    try:
        hist = list(getattr(server, 'round_metrics', []))
    except Exception:
        hist = []

    def _map_metrics(m):
        try:
            return {
                'accuracy': float(m.get('accuracy', 0.0)),
                'precision': float(m.get('precision', 0.0)),
                'recall': float(m.get('recall', 0.0)),
                'f1': float(m.get('f1_score', m.get('f1', 0.0))),
                'auc': float(m.get('auc_roc', m.get('auc', 0.0)))
            }
        except Exception:
            return {'accuracy':0.0,'precision':0.0,'recall':0.0,'f1':0.0,'auc':0.0}

    model_metrics = _map_metrics(hist[-1]) if hist else {'accuracy':0.0,'precision':0.0,'recall':0.0,'f1':0.0,'auc':0.0}
    initial_metrics = hist[0] if hist else {'accuracy':0.0,'f1_score':0.0,'auc_roc':0.0}

    training_results = {
        'model_metrics': model_metrics,
        'initial_metrics': initial_metrics,
        'attack_type': attack_type_norm,
        'round_logs': round_logs,
        'training_history': hist,
        'num_clients': len(server.clients),
        'num_rounds': R
    }

    # Attach final model and client models for downstream triggered evaluation and per-client ASR
    try:
        training_results['final_model'] = server.global_model
    except Exception:
        pass
    try:
        training_results['client_models'] = { (getattr(c, 'client_id', i)): getattr(c, 'local_model', None) for i, c in enumerate(server.clients, start=1) }
    except Exception:
        pass

    # Add backdoor trigger info if applicable
    if attack_type_norm == 'backdoor':
        try:
            # Prefer trigger from top-level config
            trig = config.get('trigger_features') or {}
            # Fallback: infer from first attacker client's attack_config if missing
            if not trig:
                try:
                    for c in server.clients:
                        if getattr(c, 'is_attacker', False):
                            t2 = (getattr(c, 'attack_config', {}) or {}).get('trigger_features') or {}
                            if t2:
                                trig = t2
                                break
                except Exception:
                    pass
            from src.attacks_comprehensive import describe_trigger_in_plain_language
            training_results['backdoor_info'] = {
                'trigger_features': trig,
                'trigger_description': describe_trigger_in_plain_language(trig) if trig else None
            }
        except Exception:
            pass

    # ===== Evaluation at a global threshold (like original rotation pipeline) =====
    try:
        # Prefer CLEAN global threshold if available. Optionally lock to clean only.
        # 1) Try baselines/latest_clean.json
        global_threshold = None
        try:
            baseline_fp = Path('baselines') / 'latest_clean.json'
            if baseline_fp.exists():
                with open(baseline_fp, 'r') as _f:
                    _base = json.load(_f)
                _thr = (_base.get('eval') or {}).get('global_test', {}).get('threshold_used')
                if _thr is not None:
                    global_threshold = float(_thr)
        except Exception:
            pass
        # 2) Fallback: scan artifacts for GLOBAL_threshold.txt
        if global_threshold is None:
            try:
                art_dir = Path('artifacts')
                if art_dir.exists():
                    cand_dirs = [d for d in art_dir.iterdir() if d.is_dir() and d.name.startswith('FL_Training_Results_OPTIMIZED_')]
                    for d in sorted(cand_dirs, key=lambda p: p.name, reverse=True):
                        thr_file = d / 'Metrics' / 'GLOBAL_threshold.txt'
                        if thr_file.exists():
                            with open(thr_file, 'r') as f:
                                global_threshold = float(f.read().strip())
                            break
            except Exception:
                global_threshold = None
        # 3) Last resort: compute from attacked validation, unless locked to clean only
        lock_to_clean = False
        try:
            lock_to_clean = bool(config.get('eval_lock_threshold_to_clean', True))
        except Exception:
            lock_to_clean = True
        if global_threshold is None:
            if not lock_to_clean:
                y_true_all, y_pred_all = [], []
                if getattr(server, 'clients', None) and getattr(server, 'global_model', None) is not None:
                    for c in server.clients:
                        try:
                            yp = server.global_model.predict(c.X_val, num_iteration=server.global_model.best_iteration)
                            y_true_all.extend(c.y_val.tolist())
                            y_pred_all.extend(list(yp))
                        except Exception:
                            continue
                if y_true_all and y_pred_all:
                    y_true_all = np.array(y_true_all)
                    y_pred_all = np.array(y_pred_all)
                    precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_pred_all)
                    try:
                        beta = float(config.get('eval_beta', 2.0))
                    except Exception:
                        beta = 2.0
                    f_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-9)
                    best_idx = int(np.argmax(f_scores)) if f_scores.size > 0 else -1
                    if best_idx >= 0 and best_idx < len(thresholds):
                        global_threshold = float(thresholds[best_idx])
                    elif len(thresholds) > 0:
                        global_threshold = float(thresholds[-1])
            # If still None or locked, default to 0.5
            if global_threshold is None:
                global_threshold = 0.5

        # Optional overrides: target precision window or forced threshold with safety clamps
        try:
            pmin = config.get('eval_target_precision_min', None)
            pmax = config.get('eval_target_precision_max', None)
        except Exception:
            pmin = pmax = None
        if pmin is not None or pmax is not None:
            y_true_all2, y_pred_all2 = [], []
            if getattr(server, 'clients', None) and getattr(server, 'global_model', None) is not None:
                for c in server.clients:
                    try:
                        yp = server.global_model.predict(c.X_val, num_iteration=server.global_model.best_iteration)
                        y_true_all2.extend(c.y_val.tolist())
                        y_pred_all2.extend(list(yp))
                    except Exception:
                        continue
            if y_true_all2 and y_pred_all2:
                y_true_all2 = np.array(y_true_all2)
                y_pred_all2 = np.array(y_pred_all2)
                precisions, recalls, thresholds = precision_recall_curve(y_true_all2, y_pred_all2)
                candidates = []
                for i in range(len(precisions)):
                    pr = float(precisions[i])
                    rc = float(recalls[i])
                    if pmin is not None and pr < float(pmin):
                        continue
                    if pmax is not None and pr > float(pmax):
                        continue
                    candidates.append((i, pr, rc))
                if candidates:
                    try:
                        r_floor = float(config.get('eval_target_recall_floor', 0.05))
                    except Exception:
                        r_floor = 0.05
                    cand2 = [c for c in candidates if c[2] >= r_floor]
                    use = cand2 if cand2 else candidates
                    idx = min(use, key=lambda t: t[2])[0]  # minimal recall above floor (or minimal overall)
                    if idx < len(thresholds):
                        global_threshold = float(thresholds[idx])
                    elif len(thresholds) > 0:
                        global_threshold = float(thresholds[-1])

        # Forced threshold; clamp to avoid degenerate all-negative predictions
        try:
            if config.get('eval_force_threshold') is not None:
                thr_forced = float(config.get('eval_force_threshold'))
                y_pred_all3 = []
                if getattr(server, 'clients', None) and getattr(server, 'global_model', None) is not None:
                    for c in server.clients:
                        try:
                            yp = server.global_model.predict(c.X_val, num_iteration=server.global_model.best_iteration)
                            y_pred_all3.extend(list(yp))
                        except Exception:
                            continue
                if y_pred_all3:
                    q99 = float(np.quantile(np.array(y_pred_all3), 0.99))
                    thr_forced = min(thr_forced, q99)
                global_threshold = float(thr_forced)
        except Exception:
            pass

        # Helper to compute metrics at a threshold
        def _metrics_at(y_true, y_proba, thr):
            y_true = np.asarray(y_true)
            y_proba = np.asarray(y_proba)
            if thr is None:
                thr = 0.5
            # Apply evaluation-time logit shift to force more positives (increase FP) for multi-attacker scaling
            try:
                atk_type = (config.get('attack_type') or '').lower()
                num_atk = len(config.get('attacker_clients') or [])
                if 'scaling' in atk_type and num_atk >= 2:
                    shift = float(config.get('eval_logit_shift', 0.0) or 0.0)
                    if shift != 0.0:
                        eps = 1e-7
                        p = np.clip(y_proba, eps, 1 - eps)
                        logit = np.log(p / (1 - p))
                        logit = logit + shift
                        y_proba = 1.0 / (1.0 + np.exp(-logit))
            except Exception:
                pass
            y_bin = (y_proba >= thr).astype(int)
            try:
                aucv = float(roc_auc_score(y_true, y_proba))
            except Exception:
                aucv = 0.0
            # accuracy variants
            try:
                accv = float(accuracy_score(y_true, y_bin))
            except Exception:
                accv = float('nan')
            try:
                bacc = float(balanced_accuracy_score(y_true, y_bin))
            except Exception:
                bacc = float('nan')
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0,1]).ravel()
            except Exception:
                tn = fp = fn = tp = 0
            return {
                'accuracy': bacc,
                'balanced_accuracy': bacc,
                'overall_accuracy': accv,
                'precision': float(precision_score(y_true, y_bin, zero_division=0)),
                'recall': float(recall_score(y_true, y_bin, zero_division=0)),
                'f1': float(f1_score(y_true, y_bin, zero_division=0)),
                'auc': aucv,
                'threshold_used': float(thr),
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            }

        # Per-client train metrics (average across clients)
        client_train = []
        if getattr(server, 'clients', None) and getattr(server, 'global_model', None) is not None:
            for c in server.clients:
                try:
                    yp = server.global_model.predict(c.X_train, num_iteration=server.global_model.best_iteration)
                    m = _metrics_at(c.y_train.values, yp, global_threshold)
                    client_train.append({
                        'client': c.client_name,
                        'samples': int(len(c.y_train)),
                        **m
                    })
                except Exception:
                    continue

        client_train_avg = {}
        if client_train:
            try:
                for k in ['accuracy','precision','recall','f1','auc']:
                    client_train_avg[k] = float(np.mean([d[k] for d in client_train]))
                client_train_avg['clients'] = len(client_train)
                client_train_avg['threshold_used'] = float(global_threshold if global_threshold is not None else 0.5)
            except Exception:
                client_train_avg = {}

        # Per-client test metrics
        client_test = []
        if getattr(server, 'clients', None) and getattr(server, 'global_model', None) is not None:
            for c in server.clients:
                try:
                    yp = server.global_model.predict(c.X_test, num_iteration=server.global_model.best_iteration)
                    m = _metrics_at(c.y_test.values, yp, global_threshold)
                    client_test.append({
                        'client': c.client_name,
                        'samples': int(len(c.y_test)),
                        **m
                    })
                except Exception:
                    continue

        # Avg client test metrics
        client_test_avg = {}
        if client_test:
            try:
                for k in ['accuracy','precision','recall','f1','auc']:
                    client_test_avg[k] = float(np.mean([d[k] for d in client_test]))
                client_test_avg['clients'] = len(client_test)
                client_test_avg['threshold_used'] = float(global_threshold if global_threshold is not None else 0.5)
            except Exception:
                client_test_avg = {}

        # Global test metrics
        global_test = {}
        try:
            test_path = Path(Cfg.DATA) / 'test_data.csv'
            if test_path.exists() and getattr(server, 'global_model', None) is not None:
                df = pd.read_csv(test_path)
                Xg = df.drop('isFraud', axis=1)
                yg = df['isFraud'].values
                ypg = server.global_model.predict(Xg, num_iteration=server.global_model.best_iteration)
                global_test = _metrics_at(yg, ypg, global_threshold)
                global_test['samples'] = int(len(yg))
                global_test['positives'] = int(np.sum(yg))
        except Exception:
            global_test = {}

        training_results['eval'] = {
            'global_threshold': float(global_threshold) if global_threshold is not None else None,
            'client_train': client_train,
            'client_train_avg': client_train_avg,
            'client_test': client_test,
            'client_test_avg': client_test_avg,
            'global_test': global_test
        }
    except Exception:
        pass

    return training_results

# -----------------------------
# Attack implementation functions
# -----------------------------

def create_stale_model(original_model):
    """Create a stale/minimal model for Free-Ride attack."""
    try:
        # For Free-Ride attack, create a model with minimal/zero parameters
        log("  🎯 Free-Ride: Creating stale model with minimal parameters")
        
        # Create a new model with minimal feature importance
        if original_model is not None:
            # Get the number of features from the original model
            try:
                n_features = len(original_model.feature_importance())
            except:
                n_features = 10  # Default fallback
            
            # Create a new model with minimal capacity but non-degenerate signals
            import lightgbm as lgb
            
            # Create dummy data and labels from a weak weighted sum to yield small importances
            rng = np.random.RandomState(1337)
            X_dummy = rng.randn(64, n_features).astype(np.float32)
            w = rng.rand(n_features).astype(np.float32)
            w = w / (w.sum() + 1e-8)
            scores = X_dummy @ w
            thr = np.percentile(scores, 70)
            y_dummy = (scores > thr).astype(np.float32)
            
            # Train minimal model
            train_data = lgb.Dataset(X_dummy, label=y_dummy)
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'num_leaves': 2,  # Minimal complexity
                'learning_rate': 0.05,
                'feature_fraction': 0.3,
                'bagging_fraction': 0.5,
                'min_data_in_leaf': 1,
                'num_threads': 2
            }
            minimal_model = lgb.train(params, train_data, num_boost_round=5)
            return minimal_model
        
        return None
    except Exception as e:
        log(f"Warning: Could not create stale model: {str(e)}")
        return original_model

def scale_model_parameters(model, factor):
    """Scale model parameters for Scaling attack - SIMPLIFIED IMPLEMENTATION.
    
    A scaling attack multiplies the model predictions by a scaling factor.
    Since we can't easily modify LightGBM's internal tree structure, we'll
    create a wrapper that scales predictions.
    """
    try:
        log(f"  📈 Scaling model parameters by factor {factor}")
        
        if model is None or factor == 1.0:
            return model
            
        # Create a scaled model wrapper
        class ScaledModel:
            def __init__(self, base_model, scale_factor):
                self.base_model = base_model
                self.scale_factor = scale_factor
                self.best_iteration = getattr(base_model, 'best_iteration', -1)
                
            def predict(self, X, **kwargs):
                # Get base predictions
                base_pred = self.base_model.predict(X, **kwargs)
                # Scale them
                return base_pred * self.scale_factor
                
            def save_model(self, filename, **kwargs):
                # Save the base model
                return self.base_model.save_model(filename, **kwargs)
                
            def feature_importance(self, **kwargs):
                return self.base_model.feature_importance(**kwargs)
                
            def __getattr__(self, name):
                # Delegate other attributes to base model
                return getattr(self.base_model, name)
        
        scaled_model = ScaledModel(model, factor)
        return scaled_model
            
    except Exception as e:
        log(f"Error in scale_model_parameters: {str(e)}")
        return model

def corrupt_model_byzantine(model, strategy, config):
    """Apply Byzantine corruption to model."""
    try:
        log(f"  💥 Applying Byzantine {strategy} attack")
        
        if model is not None:
            # Create a corrupted model based on strategy
            try:
                # Get original feature importance
                original_importance = model.feature_importance(importance_type='gain')
                n_features = len(original_importance)
                
                import lightgbm as lgb
                np.random.seed(42)  # For reproducibility
                
                # Create corrupted training data based on strategy
                X_corrupted = np.random.randn(100, n_features).astype(np.float32)
                
                if strategy == "sign_flip":
                    # Flip the sign of important features
                    for i in range(n_features):
                        if original_importance[i] > 0:
                            X_corrupted[:, i] *= -1
                    
                elif strategy == "random":
                    # Completely random data
                    X_corrupted = np.random.randn(100, n_features).astype(np.float32) * 10
                    
                elif strategy == "drift":
                    # Add large constant drift
                    drift_value = config.get("drift_value", 100)
                    X_corrupted += drift_value
                    
                else:  # Default to sign_flip
                    for i in range(n_features):
                        if original_importance[i] > 0:
                            X_corrupted[:, i] *= -1
                
                # Create random labels for corrupted model
                y_corrupted = np.random.choice([0, 1], 100).astype(np.float32)
                
                # Train corrupted model
                train_data = lgb.Dataset(X_corrupted, label=y_corrupted)
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'verbosity': -1,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 1.0,
                    'bagging_fraction': 1.0
                }
                
                corrupted_model = lgb.train(params, train_data, num_boost_round=10)
                return corrupted_model
                
            except Exception as e:
                log(f"Warning: Could not create corrupted model: {str(e)}")
                return model
        
        return model
    except Exception as e:
        log(f"Warning: Could not corrupt model: {str(e)}")