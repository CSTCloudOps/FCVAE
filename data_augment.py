import numpy as np
import torch


def missing_data_injection(x, y, z, rate):
    miss_size = int(rate * x.shape[0] * x.shape[1] * x.shape[2])
    row = torch.randint(low=0, high=x.shape[0], size=(miss_size,))
    col = torch.randint(low=0, high=x.shape[2], size=(miss_size,))
    # for i in range(miss_size):
    #     x[row[i], :, col[i]] = 0
    #     y[row[i], col[i]] = 1
    # z = torch.zeros_like(y)
    x[row, :, col] = 0
    z[row, col] = 1
    return x, y, z


def point_ano(x, y, z, rate):
    aug_size = int(rate * x.shape[0])
    id_x = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
    x_aug = x[id_x].clone()
    y_aug = y[id_x].clone()
    z_aug = z[id_x].clone()
    if x_aug.shape[1] == 1:
        ano_noise1 = torch.randint(low=1, high=20, size=(int(aug_size / 2),))
        ano_noise2 = torch.randint(
            low=-20, high=-1, size=(aug_size - int(aug_size / 2),)
        )
        ano_noise = (torch.cat((ano_noise1, ano_noise2), dim=0) / 2).to("cuda")
        x_aug[:, 0, -1] += ano_noise
        y_aug[:, -1] = torch.logical_or(y_aug[:, -1], torch.ones_like(y_aug[:, -1]))
    return x_aug, y_aug, z_aug


def seg_ano(x, y, z, rate, method):
    aug_size = int(rate * x.shape[0])
    idx_1 = torch.arange(aug_size)
    idx_2 = torch.arange(aug_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
        idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()
    time_start = torch.randint(low=7, high=x.shape[2], size=(aug_size,))  # seg start
    for i in range(len(idx_2)):
        if method == "swap":
            x_aug[i, :, time_start[i] :] = x[idx_2[i], :, time_start[i] :]
            y_aug[:, time_start[i] :] = torch.logical_or(
                y_aug[:, time_start[i] :], torch.ones_like(y_aug[:, time_start[i] :])
            )
    return x_aug, y_aug, z_aug
