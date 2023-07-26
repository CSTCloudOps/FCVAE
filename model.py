from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from CVAE import CVAE
from dataset import UniDataset
import argparse
from torch import optim
from collections import OrderedDict
import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import data_augment
from get_f1_score import best_f1, delay_f1
from Attention import EncoderLayer_selfattn

class MyVAE(LightningModule):
    def __init__(self, hparams):
        super(MyVAE, self).__init__()
        self.save_hyperparameters()
        self.hp = hparams
        self.window = hparams.window
        self.latent_dim = hparams.latent_dim
        self.hidden_dims = None
        self.step_max = 0
        self.__build_model()

    def __build_model(self):

        self.vae = CVAE(
                self.hp,
                self.hp.condition_emb_dim,
                self.latent_dim,
                1,
                self.hidden_dims,
                self.step_max,
                self.window,
                self.hp.batch_size,
            )
        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.hp.d_model,
                    self.hp.d_inner,
                    self.hp.n_head,
                    self.hp.d_model // self.hp.n_head,
                    self.hp.d_model // self.hp.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )

    def forward(self, x, mode, mask):
        x = x.view(-1, 1, self.window)
        return self.vae.forward(x, mode, mask)

    def loss(self, x, y_all, z_all, mode="train"):
        y = (y_all[:, -1]).unsqueeze(1)
        if self.hp.use_label==1:
            mask = torch.logical_not(torch.logical_or(y_all, z_all))
        else:
            mask = torch.logical_not(z_all)
        mu_x, var_x, rec_x, mu, var, loss = self.forward(
            x,
            "train",
            mask,
        )
        loss_val = loss
        if mode == "test":
            mu_x_test, recon_prob = self.forward(x, "test", z_all)
            return mu_x, var_x, recon_prob, mu_x_test
        return loss_val

    def training_step(self, data_batch, batch_idx):
        x, y_all, z_all = data_batch
        y_all2 = torch.zeros_like(y_all)
        x, y_all2, z_all = self.batch_data_augmentation(x, y_all2, z_all)
        loss_val = self.loss(x, y_all2, z_all)
        if self.trainer.strategy == "dp":
            loss_val = loss_val.unsqueeze(0)
        self.log("val_loss_train", loss_val, on_step=True, on_epoch=False, logger=True)
        output = OrderedDict(
            {
                "loss": loss_val,
            }
        )
        return output

    def validation_step(self, data_batch, batch_idx):
        x, y_all, z_all = data_batch
        y_all_wo_label = torch.zeros_like(y_all)
        loss_val = self.loss(x, y_all_wo_label, z_all)
        if self.trainer.strategy == "dp":
            loss_val = loss_val.unsqueeze(0)
        self.log("val_loss_valid", loss_val, on_step=True, on_epoch=True, logger=True)
        output = OrderedDict(
            {
                "loss": loss_val,
            }
        )
        return output

    def test_step(self, data_batch, batch_idx):
        x, y_all, z_all = data_batch
        y = (y_all[:, -1]).unsqueeze(1)
        with torch.no_grad():
            mu_x, var_x, recon_prob, mu_x_test = self.loss(x, y_all, z_all, "test")
        recon_prob = recon_prob[:, :, -1]
        output = OrderedDict(
            {
                "y": y.cpu(),
                "recon_prob": recon_prob.cpu(),
                "mu_x": mu_x[:, :, -1].cpu(),
                "mu_x_test": mu_x_test[:, :, -1].cpu(),
                "x": x[:, :, -1].cpu(),
                "var_x": var_x[:, :, -1].cpu(),
            }
        )
        return output

    def test_epoch_end(self, outputs):
        y = torch.cat(([x["y"] for x in outputs]), 0)
        recon_prob = torch.cat(([x["recon_prob"] for x in outputs]), 0)
        x = torch.cat(([x["x"] for x in outputs]), 0)
        mu_x = torch.cat(([x["mu_x"] for x in outputs]), 0)
        mu_x_test = torch.cat(([x["mu_x_test"] for x in outputs]), 0)
        var_x = torch.cat(([x["var_x"] for x in outputs]), 0)
        score = -1 * recon_prob.squeeze(1).cpu().numpy()
        label = y.squeeze(1).cpu().numpy()
        df = pd.DataFrame()
        df["x"] = x.cpu().numpy().reshape(-1)
        df["mu_x"] = mu_x.cpu().numpy().reshape(-1)
        df["mu_x_test"] = mu_x_test.cpu().numpy().reshape(-1)
        df["var_x"] = var_x.cpu().numpy().reshape(-1)
        df["y"] = y.cpu().numpy().reshape(-1)
        df["recon"] = score.reshape(-1)
        np.save('./npy/score.npy',score)
        np.save('./npy/label.npy',label)
        if self.hp.data_dir == './data/Yahoo':
            k=3
        elif self.hp.data_dir=='./data/NAB' or self.hp.data_dir=='./data/new_NAB':
            k=150
        else:
            k=7
        delay_f1_score, delay_precison, delay_recall,delay_predict = delay_f1(score, label,k)
        best_f1_socre, best_precison,best_recall,best_predict = best_f1(score, label)
        df['delay_predict'] = delay_predict
        df['best_predict'] = best_predict
        df.to_csv(
            "./csv/result.csv",
            index=False,
        )
        file_name = self.hp.save_file
        with open(file_name, "a") as f:
            f.write(
                "max f1 score is %f %f %f delay f1 score is %f %f %f\n"
                % (
                    best_f1_socre,
                    best_precison,
                    best_recall,
                    delay_f1_score,
                    delay_precison,
                    delay_recall
                )
            )

    def mydataloader(self, mode):
        dataset = UniDataset(
            self.hp.window,
            self.hp.data_dir,
            self.hp.data_name,
            mode,
            self.hp.sliding_window_size,
            data_pre_mode=self.hp.data_pre_mode,
        )
        train_sampler = None
        batch_size = self.hp.batch_size
        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
                batch_size = batch_size // self.trainer.world_size  # scale batch size
        except Exception as e:
            pass
        should_shuffle = train_sampler is None
        if mode == "valid" or mode == "test":
            should_shuffle = False
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hp.num_workers,
        )
        return loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hp.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_name", default="0efb375b-b902-3661-ab23-9a0bb799f4e3.csv", type=str
        )
        parser.add_argument("--data_dir", default="./data/AIOPS/", type=str)
        parser.add_argument("--window", default=64, type=int)
        parser.add_argument("--latent_dim", default=8, type=int)
        parser.add_argument("--only_test", default=0, type=int)
        parser.add_argument("--max_epoch", default=30, type=int)
        parser.add_argument("--batch_size", default=512, type=int)
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--learning_rate", default=0.0005, type=float)
        parser.add_argument("--sliding_window_size", default=1, type=int)
        parser.add_argument("--save_file", default="./result/base6_fix2.txt", type=str)
        parser.add_argument("--data_pre_mode", default=0, type=int)
        parser.add_argument("--missing_data_rate", default=0.01, type=float)
        parser.add_argument("--point_ano_rate", default=0.05, type=float)
        parser.add_argument("--seg_ano_rate", default=0.1, type=float)
        parser.add_argument("--eval_all", default=0, type=int)
        parser.add_argument("--condition_emb_dim", default=16, type=int)
        parser.add_argument("--atten_num", default=4, type=int)
        parser.add_argument("--d_model", default=256, type=int)
        parser.add_argument("--d_inner", default=512, type=int)
        parser.add_argument("--d_k", default=16, type=int)
        parser.add_argument("--n_head", default=8, type=int)
        parser.add_argument("--kernel_size", default=16, type=int)
        parser.add_argument("--stride", default=8, type=int)
        parser.add_argument("--mcmc_rate", default=0.2, type=float)
        parser.add_argument("--mcmc_value", default=-5, type=float)
        parser.add_argument("--mcmc_mode", default=2, type=int)#0 is rate 2 default
        parser.add_argument("--condition_mode", default=2, type=int)# 2 both local and global
        parser.add_argument("--dropout_rate", default=0.05, type=float)
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--use_label", default=1, type=int)
        return parser

    def batch_data_augmentation(self, x, y, z):
        # missing data injection
        if self.hp.point_ano_rate > 0:
            x_a, y_a, z_a = data_augment.point_ano(x, y, z, self.hp.point_ano_rate)
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)
        if self.hp.seg_ano_rate > 0:
            x_a, y_a, z_a = data_augment.seg_ano(
                x, y, z, self.hp.seg_ano_rate, method="swap"
            )
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)
        x, y, z = data_augment.missing_data_injection(
            x, y, z, self.hp.missing_data_rate
        )
        return x, y, z
