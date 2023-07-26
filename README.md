# FCVAE(ICDE 2024 Under Review)
Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective  
&bull;A new CVAE structure that using frequency as a condition.  
&bull;Using global and local frequency information makes CVAE better reconstruct normal patterns.

## Get Started
1. Install Python=3.9.13, Pytorch=1.12.1, Pytorch_lightning=1.7.7, Numpy, Pandas
2. Train and evaluate.  

```
python train.py --data_dir ./data/Yahoo  --window 48  --condition_emb_dim 64  --condition_mode 2  --save_file ./result  --gpu 0 --kernel_size 24 --stride 8 --dropout_rate 0.05
```

| Parameter | Defination |
|--------|--------|
| data_dir   |  dataset address | 
| window   | size of window   | 
|  condition_emb_dim  | dimension of condition in CVAE | 
| condition_mode   | condition class(default 2)   | 
| save_file   | address of save file   | 
| gpu   | gpu number | 
| kernel_size   | size of small window in LFM   | 
| stride   | stride in LFM when generating small windows   | 
| dropout_rate   | dropout rate   | 

## Run All Results
```
/bin/bash run_all.sh
```