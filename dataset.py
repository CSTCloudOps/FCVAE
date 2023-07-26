import torch
import torch.utils.data
import logging
import numpy as np
import pandas as pd
import os
import datapreprocess

class UniDataset(torch.utils.data.Dataset):
    def __init__(
        self, window, data_dir, data_name, mode, sliding_window_size, data_pre_mode=0
    ):
        self.window = window
        self.data_dir = data_dir
        self.data_name = data_name
        file_list = os.listdir(data_dir)
        value_all = []
        label_all = []
        missing_all = []
        self.len = 0
        self.sample_num = 0
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            df_train = df[: int(0.35 * len(df))]
            df_train = df_train.fillna(method="bfill")
            train_value = np.asarray(df_train["value"])
            train_label = np.asarray(df_train["label"])
            train_value = train_value[np.where(train_label == 0)[0]]
            train_max = train_value.max()
            train_min = train_value.min()
            if mode == "train":
                df = df[: int(0.35 * len(df))]
            elif mode == "valid":
                df = df[int(0.35 * len(df)) : int(0.5 * len(df))]
            elif mode == "test":
                df = df[int(0.5 * len(df)) :]
            timestamp, missing, (value, label) = datapreprocess.complete_timestamp(
                df["timestamp"], (df["value"], df["label"])
            )
            # unsupervised setting 
            # if mode == "train":
            #     value[np.where(label == 1)[0]] = 0
            value = value.astype(float)
            missing2 = np.isnan(value)
            missing = np.logical_or(missing, missing2).astype(int)
            label = label.astype(float)
            label[np.where(missing == 1)[0]] = np.nan
            value[np.where(missing == 1)[0]] = np.nan
            df2 = pd.DataFrame()
            df2["timestamp"] = timestamp
            df2["value"] = value
            df2["label"] = label
            df2["missing"] = missing.astype(int)
            df2 = df2.fillna(method="bfill")
            df2 = df2.fillna(0)
            df2["label"] = df2["label"].astype(int)

            if data_pre_mode == 0:
                df2["value"], *_ = datapreprocess.standardize_kpi(df2["value"])
            else:
                v = np.asarray(df2["value"])
                v = 2 * (v - train_min) / (train_max - train_min) - 1
                df2["value"] = v

            timestamp, values, labels = (
                np.asarray(df2["timestamp"]),
                np.clip(np.asarray(df2["value"]), -40, 40),
                np.asarray(df2["label"]),
            )
            values[np.where(missing == 1)[0]] = 0
            values = np.convolve(
                values,
                np.ones((sliding_window_size,)) / sliding_window_size,
                mode="valid",
            )
            timestamp = timestamp[sliding_window_size - 1 :]
            labels = labels[sliding_window_size - 1 :]
            missing = missing[sliding_window_size - 1 :]
            value_all.append(values)
            label_all.append(labels)
            missing_all.append(missing)
            self.sample_num += max(len(values) - window + 1, 0)
        self.samples, self.labels, self.miss_label = self.__getsamples(
            value_all, label_all, missing_all
        )

    def __getsamples(self, values, labels, missing):
        X = torch.zeros((self.sample_num, 1, self.window))
        Y = torch.zeros((self.sample_num, self.window))
        Z = torch.zeros((self.sample_num, self.window))
        i = 0
        for cnt in range(len(values)):
            v = values[cnt]
            l = labels[cnt]
            m = missing[cnt]
            for j in range(len(v) - self.window + 1):
                X[i, 0, :] = torch.from_numpy(v[j : j + self.window])
                Y[i, :] = torch.from_numpy(np.asarray(l[j : j + self.window]))
                Z[i, :] = torch.from_numpy(np.asarray(m[j : j + self.window]))
                i += 1
        return (X, Y, Z)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :], self.miss_label[idx, :]]
        return sample
