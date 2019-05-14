import torch
import numpy as np


## Stolen from https://github.com/ethanfetaya/NRI/blob/master/utils.py

def kl_categorical(log_prior, num_atoms, eps=1e-16):
    def f(preds):
        return _kl_categorical(preds, log_prior, num_atoms, eps)

    return f


def kl_categorical_uniform(num_atoms, num_edge_types, add_const=False, eps=1e-16):
    def f(preds):
        return _kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const, eps)

    return f


def nll_gaussian(variance, add_const=False):
    def f(preds, target):
        return _nll_gaussian(preds, target, variance, add_const)

    return f


def _kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def _kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                            eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def _nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))
