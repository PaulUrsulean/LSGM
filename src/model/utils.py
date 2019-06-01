import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Gumbel


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.to(logits.device)
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.to(y_soft.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def my_softmax(input, axis=1):
    # From https://github.com/ethanfetaya/NRI/blob/master/utils.py
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def encode_onehot(labels):
    """
    From https://github.com/ethanfetaya/NRI/blob/master/utils.py
    :param labels: 
    :return: 
    """

    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)

    return labels_onehot


def gen_fully_connected(n_elements, device=None):
    # From https://github.com/ethanfetaya/NRI/blob/master/utils.py
    # Generate off-diagonal interaction graph
    off_diag = np.ones([n_elements, n_elements]) - np.eye(n_elements)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    if device:
        rel_rec, rel_send = rel_rec.to(device), rel_send.to(device)

    return rel_rec, rel_send


def node2edge(m, adj_rec=None, adj_send=None):
    """
    Calculates edge embeddings
    :param m: Tensor with shape (SAMPLES, OBJECTS, FEATURES)
    :param adj_rec:
    :param adj_send:
    :return:
    """
    outgoing = torch.matmul(adj_send, m)
    incoming = torch.matmul(adj_rec, m)
    return torch.cat([outgoing, incoming], dim=2)


def edge2node(m, adj_rec, adj_send):
    """
    Performs accumulation of message passing by summing over connected edges for each node
    :param x: tensor with shape (N_OBJ, N_HIDDEN)
    :param adj_rec:
    :param adj_send:
    :return:
    """
    incoming = torch.matmul(adj_rec.t(), m)
    return incoming / incoming.size(1)


def load_models(enc: torch.nn.Module, dec: torch.nn.Module, config: dict):
    models_path = config['training']['load_path']
    path = Path(models_path).parent / "models"

    # Find different models for each epoch
    max_epoch = -1
    for f in os.listdir(path):
        epoch = int(f.split("_epoch")[-1].split(".pt")[0])
        max_epoch = max(epoch, max_epoch)

    if max_epoch == -1:
        raise FileNotFoundError(f"No models found under {models_path}")

    encoder_file = path / f"encoder_epoch{max_epoch}.pt"
    decoder_file = path / f"decoder_epoch{max_epoch}.pt"
    enc.load_state_dict(torch.load(encoder_file))
    dec.load_state_dict(torch.load(decoder_file))

    print(f"Loaded encoder {encoder_file} and decoder {decoder_file}")
    config['training']['load_path'] = None
    return enc, dec


def load_lstm_models(rnn: torch.nn.Module, config:dict):

    models_path = config['training']['load_path']
    path = Path(models_path).parent / "models"

    # Find different models for each epoch
    max_epoch=-1
    for f in os.listdir(path):
        epoch = int(f.split("_epoch")[-1].split(".pt")[0])
        max_epoch = max(epoch, max_epoch)

    if max_epoch == -1:
        raise FileNotFoundError(f"No models found under {models_path}")

    rnn_file = path / f"rnn_epoch{max_epoch}".pt
    rnn.load_state_dict(torch.load(rnn_file))

    print(f"Loaded rnn {rnn_file}")
    config['training']['load_path']=None
    return rnn




def nll():
    pass


def kl():
    pass
