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


class RecVAE(nn.Module):
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
        self.old_p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, x, calculate_loss = True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z)
        
        if calculate_loss:
            BCE = -torch.mean(torch.sum(F.log_softmax(x_pred, 1) * x, -1))
            KLD = (log_norm_pdf(z, mu, logvar) - self.prior(x, z)).sum(dim=-1).mean()
            return BCE + self.anneal * KLD
        else:
            return self.decode(z), mu, logvar, z
    
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


class DisenVAE(nn.Module):
    # def __init__(self, M, K, D, tau, dropout):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(DisenVAE, self).__init__()

        self.M = p_dims[0]
        self.H = D * 3
        self.D = p_dims[-1]
        self.K = K
        self.tau = tau

        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D * 2)
        )
        self.items = Parameter(torch.Tensor(self.M, self.D))
        self.cores = Parameter(torch.Tensor(self.K, self.D))
        self.drop = nn.Dropout(dropout)

        init.xavier_normal_(self.items)
        init.xavier_normal_(self.cores)


    def cluster(self):
        items = F.normalize(self.items, dim=1)      # M * D
        cores = F.normalize(self.cores, dim=1)      # K * D
        cates = torch.mm(items, cores.t()) / self.tau
        cates = F.softmax(cates, dim=1)             # M * K
        return items, cores, cates


    def encode(self, X, cates):
        n = X.shape[0]
        X = self.drop(X)
        X = X.view(n, 1, self.M) *  \
            cates.t().expand(n, self.K, self.M)     # n * K * M
        X = X.reshape(n * self.K, self.M)           # (n * K) * M
        h = self.encoder(X)                         # (n * K) * D * 2
        mu, logvar = h[:, :self.D], h[:, self.D:]   # (n * k) * D
        return mu, logvar


    def decode(self, z, items, cates):
        n = z.shape[0] // self.K
        z = F.normalize(z, dim=1)                   # (n * K) * D
        logits = torch.mm(z, items.t()) / self.tau  # (n * K) * M
        probs = torch.exp(logits)                   # (n * K) * M
        probs = torch.sum(probs.view(n, self.K, self.M) * \
                cates.t().expand(n, self.K, self.M), dim=1)
        logits = torch.log(probs)
        logits = F.log_softmax(logits, dim=1)
        return logits


    def sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu


    def forward(self, X, A):
        items, cores, cates = self.cluster()
        mu, logvar = self.encode(X, cates)
        z = self.sample(mu, logvar)
        logits = self.decode(z, items, cates)
        return logits, mu, logvar, None, None, None
    

    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits) + anneal * kl_loss(X_mu, X_logvar)


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
