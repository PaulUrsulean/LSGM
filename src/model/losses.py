import torch
import numpy as np


def vae_loss(predictions,
             targets,
             edge_probs,
             n_atoms,
             n_edge_types,
             device=None,
             log_prior=None,
             add_const=False,
             eps=1e-16,
             beta=1.0,
             prediction_variance=5e-5):
    """
    Calculate complete loss including both negative log-likelihood and KL-divergence.
    Additionally enables a flexible emphasis on the prior via the beta parameter.
    :param predictions: Predicted future states
    :param targets: Ground truth future states
    :param edge_probs: Latent edge-types probabilities
    :param n_atoms: Number of atoms in the simulation
    :param n_edge_types: Number of different edge-types
    :param device: device to which log-prior is moved to calculate loss more efficiently
    :param log_prior: List of probabilities about the prior (e.g. [0.95, .05]. Sums up to 1!). None for uniform prior.
    :param add_const: Add constant to nll-loss (never used in original implementation)
    :param eps: Small value to avaid numerical issues
    :param beta: Factor for the KL-divergence. Choose value > 1 for stronger emphasis on the latent prior.
    :param prediction_variance: Variance of the future predictions output, which is a normal distribution.
    :return: Tuple of (total loss, nll, kl-div)
    """
    nll = nll_gaussian(predictions, targets, prediction_variance, add_const)
    if log_prior is None:
        kl_div = kl_categorical_uniform(edge_probs, n_atoms, n_edge_types, add_const=add_const, eps=eps)
    else:
        log_prior = torch.Tensor(log_prior).to(device or torch.device('cpu'))
        kl_div = kl_categorical(edge_probs, log_prior, n_atoms, eps=eps)
    return nll + kl_div, nll, kl_div


"""
Calculations below taken from https://github.com/ethanfetaya/NRI/blob/master/utils.py with adaptions
"""


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))
