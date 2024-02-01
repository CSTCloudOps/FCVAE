import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from Attention import EncoderLayer_selfattn
import pywt
class CVAE(nn.Module):
    def __init__(
        self,
        hp,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "C",
    ):
        super(CVAE, self).__init__()
        self.hp = hp
        self.num_iter = 0
        self.step_max = 0
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        modules = []
        in_channels = self.hp.window + 2*self.hp.condition_emb_dim
        self.hidden_dims = [100, 100]
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
            nn.Softplus(),
        )
        modules = []
        self.decoder_input = nn.Linear(
            self.hp.latent_dim + 2 * self.hp.condition_emb_dim, self.hidden_dims[-1]
        )
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    nn.Tanh(),
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.hp.window),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        self.fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.window),
            nn.Softplus()
        )
        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.hp.d_model,
                    self.hp.d_inner,
                    self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )
        self.emb_local = nn.Sequential(
            nn.Linear(2+self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model,self.hp.condition_emb_dim),
            nn.Tanh(),
        )
        self.dropout =nn.Dropout(self.hp.dropout_rate)
        self.emb_global = nn.Sequential(
            nn.Linear(self.hp.window,self.hp.condition_emb_dim),
            nn.Tanh(),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        return [mu, var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 1, self.hidden_dims[0])
        result = self.decoder(result)
        mu_x = self.fc_mu_x(result)
        var_x = self.fc_var_x(result)
        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y):
        if mode == "train" or mode == "valid":
            condition = self.get_conditon(input)
            condition = self.dropout(condition)
            mu, var = self.encode(torch.cat((input, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            rec_x = self.reparameterize(mu_x, var_x)
            loss = self.loss_func(mu_x, var_x, input, mu, var, y, z)
            return [mu_x, var_x, rec_x, mu, var, loss]
        else:
            y = y.unsqueeze(1)
            return self.MCMC2(input)

    def get_conditon(self, x):
        x_g = x
        f_global = torch.fft.rfft(x_g[:,:,:-1],dim=-1)
        f_global = torch.cat((f_global.real,f_global.imag),dim=-1)
        f_global = self.emb_global(f_global)
        x_g = x_g.view(x.shape[0], 1, 1, -1)
        x_l = x_g.clone()
        x_l[:,:,:,-1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.hp.stride),
        )
        unfold_x = unfold(x_l)
        unfold_x = unfold_x.transpose(1, 2)
        f_local = torch.fft.rfft(unfold_x, dim=-1)
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)
        f_local = self.emb_local(f_local)
        for enc_layer in self.atten:
            f_local, enc_slf_attn = enc_layer(f_local)
        f_local = self.out_linear(f_local)
        f_local = f_local[:, -1, :].unsqueeze(1)
        output = torch.cat((f_global,f_local),-1)
        return output
    
    def MCMC2(self, x):
        condition = self.get_conditon(x)
        origin_x = x.clone()
        for i in range(10):
            mu, var = self.encode(torch.cat((x, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            recon = -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)
            temp = (
                torch.from_numpy(np.percentile(recon.cpu(), self.hp.mcmc_rate, axis=-1))
                .unsqueeze(2)
                .repeat(1, 1, self.hp.window)
            ).to("cuda")
            if(self.hp.mcmc_mode==0):
                l = (temp < recon).int()
                x = mu_x * (1 - l) + origin_x * l
            if(self.hp.mcmc_mode==1):
                l = (self.hp.mcmc_value < recon).int()
                x = origin_x * l+mu_x * (1 - l)
            if(self.hp.mcmc_mode==2):
                l = torch.ones_like(origin_x)
                l[:,:,-1]=0
                x = origin_x*l +(1-l)*mu_x
        prob_all = 0
        mu, var = self.encode(torch.cat((x, condition), dim=2))
        for i in range(128):
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            prob_all += -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)
        return x, prob_all / 128

    def loss_func(self, mu_x, var_x, input, mu, var, y, z, mode="nottrain"):
        if mode == "train":
            self.num_iter += 1
            self.num_iter = self.num_iter % 100
        kld_weight = 0.005
        mu_x = mu_x.squeeze(1)
        var_x = var_x.squeeze(1)
        input = input.squeeze(1)
        recon_loss = torch.mean(
            0.5
            * torch.mean(
                y * (torch.log(var_x) + (input - mu_x) ** 2 / var_x), dim=1
            ),
            dim=0,
        )
        m = (torch.sum(y, dim=1, keepdim=True) / self.hp.window).repeat(
            1, self.hp.latent_dim
        )
        kld_loss = torch.mean(
            0.5
            * torch.mean(m * (z**2) - torch.log(var) - (z - mu) ** 2 / var, dim=1),
            dim=0,
        )
        if self.loss_type == "B":
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recon_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        elif self.loss_type == "C":
            loss = recon_loss + kld_loss
        elif self.loss_type == "D":
            loss = recon_loss + self.num_iter / 100 * kld_loss
        else:
            raise ValueError("Undefined loss type.")
        return loss
