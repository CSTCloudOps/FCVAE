# FCVAE WWW 2024
Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective  
&bull;A new CVAE structure that using frequency as a condition.  
&bull;Using global and local frequency information makes CVAE better reconstruct normal patterns.

## Disclaimer
The resources, including code, data, and model weights, associated with this project are restricted for academic research purposes only and cannot be used for commercial purposes. The content produced by any version of FCVAE is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project. This project does not accept any legal liability for the content of the model output, nor does it assume responsibility for any losses incurred due to the use of associated resources and output results.

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
| condition_emb_dim  | dimension of condition in CVAE | 
| condition_mode   | condition class(default 2)   | 
| save_file   | address of save file   | 
| gpu   | gpu number | 
| kernel_size   | size of small window in LFM   | 
| stride   | stride in LFM when generating small windows   | 
| dropout_rate   | dropout rate   | 
| use_label   | 1:supervised 0:unsupervised   | 
| latent_dim  | dimension of latent space   | 
| max_epoch  | training epoches   | 
| batch_size  | batch_size   | 
| learning_rate  | learning rate   | 
| data_pre_mode  | datapreprocessing mode  | 
| missing_data_rate  | missing data injection rate  | 
| mcmc_mode | default:2  | 

## Run All Results
```
/bin/bash run_all.sh
```