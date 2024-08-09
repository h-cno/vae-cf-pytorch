import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)
        
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.weights) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    
def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD
            
            
# class ContrastVAE_VD(ContrastVAE):

#     def __init__(self, p_dims, q_dims=None, dropout=0.5):
#         super(MultiVAE, self).__init__()
#         self.p_dims = p_dims
#         if q_dims:
#             assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
#             assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
#             self.q_dims = q_dims
#         else:
#             self.q_dims = p_dims[::-1]

#         # Last dimension of q- network is for mean and variance
#         temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
#         self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
#             d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
#         self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
#             d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
#         self.drop = nn.Dropout(dropout)
#         self.init_weights()
    
#     def forward(self, input, input_aug):
#         mu, logvar = self.encode(input)
#         mu_aug, logvar_aug = self.encode(input_aug)
        
#         z = self.reparameterize(mu, logvar)
#         z_aug = self.reparameterize(mu_aug, logvar_aug)
        
#         return self.decode(z), self.decode(z_aug), mu, mu_aug, logvar, logvar_aug
    
#     def encode(self, input):
#         h = F.normalize(input)
#         h = self.drop(h)
        
#         for i, layer in enumerate(self.q_layers):
#             h = layer(h)
#             if i != len(self.q_layers) - 1:
#                 h = F.tanh(h)
#             else:
#                 mu = h[:, :self.q_dims[-1]]
#                 logvar = h[:, self.q_dims[-1]:]
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std, alpha = self.latent_dropout_VD(torch.exp(0.5*logvar))
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
    
#     def decode(self, z):
#         h = z
#         for i, layer in enumerate(self.p_layers):
#             h = layer(h)
#             if i != len(self.p_layers) - 1:
#                 h = F.tanh(h)
#         return h

#     def init_weights(self):
#         for layer in self.q_layers:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0/(fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for Biases
#             layer.bias.data.normal_(0.0, 0.001)
        
#         for layer in self.p_layers:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0/(fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for Biases
#             layer.bias.data.normal_(0.0, 0.001)




# def loss_function_VD(recon_x, recon_x_aug, x, x_aug, mu, mu_aug, logvar, logvar_aug, alpha):
#     recons_loss = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
#     recons_loss_aug = -torch.mean(torch.sum(F.log_softmax(recon_x_aug, 1) * x_aug, -1))
    
#     kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
#     kld_loss_aug = -0.5 * torch.mean(torch.sum(1 + logvar_aug - mu_aug.pow(2) - logvar_aug.exp(), dim=1))
    
#     user_representation = torch.sum(z, dim=1)
#     user_representation_aug = torch.sum(z_aug, dim=1)
#     contrastive_loss = self.cl_criterion(user_representation, user_representation_aug)
    
#     adaptive_alpha_loss = priorKL(alpha)
    
#     return recons_loss + recons_loss_aug + kld_weight * (kld_loss + kld_loss_aug) + self.args.latent_clr_weight * contrastive_loss+ adaptive_alpha_loss
    
    
class MultiBetaVAE(nn.Module):
    """
    Container module for Multi-VAE with beta-distribution.
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiBetaVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        alpha, beta = self.encode(input)
        z = self.reparameterize(alpha, beta)
        return self.decode(z), alpha, beta
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            # print(h)
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                alpha = F.softplus(h[:, :self.q_dims[-1]]) + 1e-10  
                beta = F.softplus(h[:, self.q_dims[-1]:]) + 1e-10 
                
                print(alpha.shape)
                print("-----")
        return alpha, beta

    def reparameterize(self, alpha, beta):
        dist = Beta(alpha, beta)
        return(dist.rsample())
        # if self.training:
        #     std = torch.exp(0.5 * logvar)
        #     eps = torch.randn_like(std)
        #     return eps.mul(std).add_(mu)
        # else:
        #     return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    
def loss_function_beta(recon_x, x, alpha, beta, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = torch.sum(Beta(alpha, beta).log_prob(alpha / (alpha + beta)) - torch.log(1.0 / (alpha + beta)))
    print(KLD)
    return BCE + anneal * KLD