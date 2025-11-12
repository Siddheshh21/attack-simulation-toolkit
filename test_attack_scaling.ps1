# Comprehensive attack scaling test suite
# Tests different combinations of attackers and flip percentages

$results = @()

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ATTACK SCALING TEST SUITE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 1: 2 attackers, flip=0.9 (baseline)
Write-Host "`n[TEST 1] 2 attackers, flip=0.9 (baseline)" -ForegroundColor Yellow
$input1 = @"
1
2,4
0.9
"@
$output1 = $input1 | .venv\Scripts\python.exe src\interactive_attack_tester.py 2>&1 | Out-String
$results += @{Test="2atk_flip0.9"; Output=$output1}
Start-Sleep -Seconds 2

# Test 2: 2 attackers, flip=0.7
Write-Host "`n[TEST 2] 2 attackers, flip=0.7" -ForegroundColor Yellow
$input2 = @"
1
3,5
0.7
"@
$output2 = $input2 | .venv\Scripts\python.exe src\interactive_attack_tester.py 2>&1 | Out-String
$results += @{Test="2atk_flip0.7"; Output=$output2}
Start-Sleep -Seconds 2

# Test 3: 1 attacker, flip=0.8
Write-Host "`n[TEST 3] 1 attacker, flip=0.8" -ForegroundColor Yellow
$input3 = @"
1
2
0.8
"@
$output3 = $input3 | .venv\Scripts\python.exe src\interactive_attack_tester.py 2>&1 | Out-String
$results += @{Test="1atk_flip0.8"; Output=$output3}
Start-Sleep -Seconds 2

# Test 4: 2 attackers, flip=0.3 (lower bound)
Write-Host "`n[TEST 4] 2 attackers, flip=0.3 (lower bound)" -ForegroundColor Yellow
$input4 = @"
1
1,3
0.3
"@
$output4 = $input4 | .venv\Scripts\python.exe src\interactive_attack_tester.py 2>&1 | Out-String
$results += @{Test="2atk_flip0.3"; Output=$output4}
Start-Sleep -Seconds 2

# Test 5: 1 attacker, flip=0.6
Write-Host "`n[TEST 5] 1 attacker, flip=0.6" -ForegroundColor Yellow
$input5 = @"
1
4
0.6
"@
$output5 = $input5 | .venv\Scripts\python.exe src\interactive_attack_tester.py 2>&1 | Out-String
$results += @{Test="1atk_flip0.6"; Output=$output5}

# Extract and display results
Write-Host "`n`n========================================" -ForegroundColor Cyan
Write-Host "TEST RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

foreach ($result in $results) {
    Write-Host "`n--- $($result.Test) ---" -ForegroundColor Green
    
    # Extract key metrics using regex
    if ($result.Output -match "Delta\s+-> Accuracy:([+-]?\d+\.\d+) \(([+-]?\d+\.\d+)%\).*Recall:([+-]?\d+\.\d+) \(([+-]?\d+\.\d+)%\)") {
        $accDrop = $matches[2]
        $recDrop = $matches[4]
        Write-Host "  Accuracy drop: $accDrop%" -ForegroundColor $(if ([double]$accDrop -ge -25 -and [double]$accDrop -le -12) {"Green"} else {"Red"})
        Write-Host "  Recall drop: $recDrop%" -ForegroundColor $(if ([double]$recDrop -ge -9 -and [double]$recDrop -le -8) {"Green"} else {"Red"})
    }
    
    if ($result.Output -match "Attacked -> Accuracy:\d+\.\d+ \| Prec:(\d+\.\d+)") {
        $prec = $matches[1]
        Write-Host "  Precision: $prec" -ForegroundColor $(if ([double]$prec -ge 0.15 -and [double]$prec -le 0.25) {"Green"} else {"Red"})
    }
    
    if ($result.Output -match "\[DYNAMIC SCALING\] flip=(\d+\.\d+), attackers=(\d+), factor=(\d+\.\d+), multiplier=(\d+\.\d+)") {
        Write-Host "  Scaling: flip=$($matches[1]), atk=$($matches[2]), mult=$($matches[4])" -ForegroundColor Cyan
    }
}

Write-Host "`n`nAll tests completed!" -ForegroundColor Green
