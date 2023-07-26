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
| data_dir   | 数据集地址  | 
| window   | 窗口长度   | 
|  condition_emb_dim  | conditon维度   | 
| condition_mode   | condition类别，默认2   | 
| save_file   | 结果保存文件地址   | 
| gpu   | gpu卡号  | 
| kernel_size   | LFM中小窗口长度   | 
| stride   | LFM生成小窗口的步长   | 
| dropout_rate   | Dropout比例   | 

## Run All Results
```
/bin/bash run_all.sh
```