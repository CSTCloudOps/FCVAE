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
        condition_emb_dim,
        latent_dim,
        in_channels,
        hidden_dims,
        step_max,
        window,
        batch_size,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "C",
    ):
        super(CVAE, self).__init__()
        self.hp = hp
        self.condition_emb_dim = 2*self.hp.condition_emb_dim
        self.latent_dim = latent_dim
        modules = []
        self.num_iter = 0
        self.step_max = step_max
        self.window = window
        self.batch_size = batch_size
        self.step_now = 0
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        in_channels = window + self.condition_emb_dim
        if hidden_dims is None:
            hidden_dims = [100, 100]
        self.hidden_dims = hidden_dims
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.now_dim = int(self.window / (2 ** len(hidden_dims)))
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.Softplus(),
        )

        modules = []

        self.decoder_input = nn.Linear(
            latent_dim + self.condition_emb_dim, hidden_dims[-1]
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.Tanh(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], self.window),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.fc_mu_x = nn.Linear(self.window, self.window)
        self.fc_var_x = nn.Sequential(
            nn.Linear(self.window, self.window),
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
        self.in_linear = nn.Sequential(
            nn.Linear(2+self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.condition_emb_dim = self.condition_emb_dim//2
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model,self.condition_emb_dim),
            nn.Tanh(),
        )
        self.dropout =nn.Dropout(self.hp.dropout_rate)
        self.emb = nn.Sequential(
            nn.Linear(self.hp.window,self.condition_emb_dim),
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

    def forward(self, input, mode, y_all):
        if mode == "train" or mode == "valid":
            condition = self.get_conditon(input)
            condition = self.dropout(condition)
            mu, var = self.encode(torch.cat((input, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            rec_x = self.reparameterize(mu_x, var_x)
            loss = self.loss_func(mu_x, var_x, input, mu, var, y_all, z)
            return [mu_x, var_x, rec_x, mu, var, loss]
        else:
            y_all = y_all.unsqueeze(1)
            x = input
            return self.MCMC2(x)

    def get_conditon(self, x):
        x_c =x
        f_global = torch.fft.rfft(x_c[:,:,:-1],dim=-1)
        f_global = torch.cat((f_global.real,f_global.imag),dim=-1)
        f_global = self.emb(f_global)
        x_c = x_c.view(x.shape[0], 1, 1, -1)
        x_c_l = x_c.clone()
        x_c_l[:,:,:,-1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.hp.stride),
        )
        unfold_x = unfold(x_c_l)
        unfold_x = unfold_x.transpose(1, 2)
        freq = torch.fft.rfft(unfold_x, dim=-1)
        #np.save('./npy/smallwindowfrq_{}.npy'.format(self.hp.data_dir[7:]),(torch.abs(freq)).cpu().detach().numpy())
        freq = torch.cat((freq.real, freq.imag), dim=-1)

        enc_output = self.in_linear(freq)
        for enc_layer in self.atten:
            enc_output, enc_slf_attn = enc_layer(enc_output)
        # np.save('./npy/atten_{}.npy'.format(self.hp.data_dir[7:]),enc_slf_attn.cpu().detach().numpy())
        # np.save('./npy/origin_{}.npy'.format(self.hp.data_dir[7:]),x.cpu().detach().numpy())
        # np.save('./npy/smallwindow_{}.npy'.format(self.hp.data_dir[7:]),unfold_x.cpu().detach().numpy())
        enc_output = self.out_linear(enc_output)
        f_local = enc_output[:, -1, :].unsqueeze(1)

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
                .repeat(1, 1, self.window)
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

    def loss_func(self, mu_x, var_x, input, mu, var, y_all, z, mode="nottrain"):
        if mode == "train":
            self.num_iter += 1
            self.num_iter = self.num_iter % 100
        kld_weight = 0.005
        mu_x = mu_x.squeeze(1)
        var_x = var_x.squeeze(1)
        input = input.squeeze(1)
        w = torch.zeros_like(mu_x)
        w[:,-1] = 5
        recon_loss = torch.mean(
            0.5
            * torch.mean(
                y_all * (torch.log(var_x) + (input - mu_x) ** 2 / var_x), dim=1
            ),
            dim=0,
        )
        m = (torch.sum(y_all, dim=1, keepdim=True) / self.window).repeat(
            1, self.latent_dim
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
