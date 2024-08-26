from recbole.model.abstract_recommender import MultiVAE
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
import torch
import torch.nn as nn


class StochasticMultiVAE(MultiVAE):

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
