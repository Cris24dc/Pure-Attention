## Least squares && Full Batch Gradient Descent

```bash
(.env) bloodybear@asus-tuf:~/Pure-Attention/tests/lstsq$ python3 main.py 
Ideal Bias: 2.06786
Epoch 0, MSE Loss: 5.460640
Epoch 1000, MSE Loss: 4.636060
Epoch 2000, MSE Loss: 4.131331
Epoch 3000, MSE Loss: 3.787939
Epoch 4000, MSE Loss: 3.484987
Epoch 5000, MSE Loss: 3.133950
Epoch 6000, MSE Loss: 2.856797
Epoch 7000, MSE Loss: 2.638173
Epoch 8000, MSE Loss: 2.330605
Epoch 9000, MSE Loss: 2.038079
Epoch 10000, MSE Loss: 1.819790
Epoch 11000, MSE Loss: 1.629321
Epoch 12000, MSE Loss: 1.480254
Epoch 13000, MSE Loss: 1.357425
Epoch 14000, MSE Loss: 1.215323
Epoch 15000, MSE Loss: 1.049851
Epoch 16000, MSE Loss: 0.927404
Epoch 17000, MSE Loss: 0.834543
Epoch 18000, MSE Loss: 0.759341
Epoch 19000, MSE Loss: 0.696765
Epoch 20000, MSE Loss: 0.638743
Epoch 21000, MSE Loss: 0.602109
Epoch 22000, MSE Loss: 0.582528
Epoch 23000, MSE Loss: 0.547974
Epoch 24000, MSE Loss: 0.550523
Epoch 25000, MSE Loss: 0.549002
Epoch 26000, MSE Loss: 0.559642
Epoch 27000, MSE Loss: 0.575223
Epoch 28000, MSE Loss: 0.586983
Epoch 29000, MSE Loss: 0.600909
------------------------------------------------------------
Param           | CUDA            | NumPy           | Diff           
------------------------------------------------------------
Bias            | 2.06803         | 2.06786         | 0.00016        
W[0]            | 0.78879         | 0.85238         | 0.06359        
W[1]            | 0.15523         | 0.12238         | 0.03285        
W[2]            | -0.06911        | -0.30512        | 0.23600        
W[3]            | 0.05205         | 0.37113         | 0.31908        
W[4]            | -0.02165        | -0.00230        | 0.01935        
------------------------------------------------------------
```

```bash
Ideal Bias: 2.06786
Epoch 0, MSE Loss: 6.295711
Epoch 1000, MSE Loss: 5.310550
Epoch 2000, MSE Loss: 4.642980
Epoch 3000, MSE Loss: 4.105417
Epoch 4000, MSE Loss: 3.686285
Epoch 5000, MSE Loss: 3.350644
Epoch 6000, MSE Loss: 3.075618
Epoch 7000, MSE Loss: 2.758446
Epoch 8000, MSE Loss: 2.399651
Epoch 9000, MSE Loss: 2.133004
Epoch 10000, MSE Loss: 1.923771
Epoch 11000, MSE Loss: 1.761896
Epoch 12000, MSE Loss: 1.632480
Epoch 13000, MSE Loss: 1.463961
Epoch 14000, MSE Loss: 1.314979
Epoch 15000, MSE Loss: 1.190136
Epoch 16000, MSE Loss: 1.061759
Epoch 17000, MSE Loss: 0.920748
Epoch 18000, MSE Loss: 0.812634
Epoch 19000, MSE Loss: 0.774354
Epoch 20000, MSE Loss: 0.816480
Epoch 21000, MSE Loss: 0.878329
Epoch 22000, MSE Loss: 0.881069
Epoch 23000, MSE Loss: 0.803440
Epoch 24000, MSE Loss: 0.756682
Epoch 25000, MSE Loss: 0.719891
Epoch 26000, MSE Loss: 0.675850
Epoch 27000, MSE Loss: 0.676250
Epoch 28000, MSE Loss: 0.691215
Epoch 29000, MSE Loss: 0.689571
------------------------------------------------------------
Param           | CUDA            | NumPy           | Diff           
------------------------------------------------------------
Bias            | 2.06706         | 2.06786         | 0.00081        
W[0]            | 1.45465         | 0.85238         | 0.60227        
W[1]            | 0.14361         | 0.12238         | 0.02123        
W[2]            | -1.40897        | -0.30512        | 1.10385        
W[3]            | 1.32651         | 0.37113         | 0.95537        
W[4]            | -0.00077        | -0.00230        | 0.00153        
W[5]            | 0.00430         | -0.03662        | 0.04092        
W[6]            | -0.83965        | -0.89664        | 0.05699        
W[7]            | -0.87633        | -0.86893        | 0.00740        
------------------------------------------------------------
```
