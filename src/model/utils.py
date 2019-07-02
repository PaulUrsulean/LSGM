import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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

    max_epoch = find_latest_checkpoint(path)

    if max_epoch == -1:
        raise FileNotFoundError(f"No models found under {models_path}")

    load_weights(enc, path / f"encoder_epoch{max_epoch}.pt")
    load_weights(dec, path / f"decoder_epoch{max_epoch}.pt")

    print(f"Loaded encoder and decoder from path {path} and epoch {max_epoch}.")
    config['training']['load_path'] = None
    return enc, dec


def find_latest_checkpoint(path):
    """
    Looks for all .pt files in the folder and returns the highest seen epoch
    :param path: Path to folder where models are stored
    :return: epoch as int
    """

    max_epoch = -1
    for f in os.listdir(path):

        # Skip if not a saved model
        if not f.endswith(".pt") or "_epoch" not in f:
            continue

        epoch = int(f.split("_epoch")[-1].split(".pt")[0])
        max_epoch = max(epoch, max_epoch)
    return max_epoch


def load_weights(model, path):
    model.load_state_dict(torch.load(path))


def nll():
    pass


def kl():
    pass


def edges_to_adj(edges, n_atoms):
    n_samples = edges.size(0)
    indices = get_offdiag_indices(n_atoms)
    n_edge_types = edges.size(-1)

    graphs = np.empty((n_samples, n_edge_types, n_atoms, n_atoms))

    for sample in range(n_samples):
        for edge_type in range(n_edge_types):
            graph = edges[sample, :, edge_type]
            fully_connected = torch.zeros(n_atoms * n_atoms)
            fully_connected[indices] = graph
            adjacency_matrix = fully_connected.view(n_atoms, n_atoms).detach().numpy()
            graphs[sample, edge_type, :, :] = adjacency_matrix

    return graphs


def id_2_loc(i):
    """
    Remove when real function is provided
    :param i:
    :return:
    """
    lat = np.random.rand() * 8 + 36
    lon = np.random.rand() * 8 - 3.9
    return (lon, lat)


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def cosine_dist(v1, v2):
    return 1 - (v1 @ v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)

def dot_product(v1, v2):
    return v1 @ v2


def cosine_signature(X, b, r):
    N, D = X.shape
    random_projections = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D), size=b*r)
    
    hash_transform = lambda x: (x>=0).astype(np.int32) * 2 - 1
    return hash_transform(random_projections @ X.T)

def euclidean_signature(X, b, r, w=1):
    N, D = X.shape
    random_projections = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D), size=b*r)
    bias = np.random.uniform(0, w, size=b*r)
    
    hash_transform = lambda x: np.floor(x)
    return hash_transform((random_projections @ X.T + bias[np.newaxis].T) / w)

def dot_signature(X, b, r):
    N, D = X.shape
    normalized = X/np.linalg.norm(X, axis=1).max()
    augment_col = np.sqrt(1 - np.linalg.norm(normalized, axis=1))[np.newaxis].T
    augmented = np.concatenate((normalized, augment_col), axis=1)
    random_vectors = np.random.multivariate_normal(mean=np.zeros(D+1), cov=np.eye(D+1), size=b*r)
    
    hash_transform = lambda x: (x>=0).astype(np.int32) * 2 - 1
    return hash_transform(random_vectors @ augmented.T)


def LSH(X, b=8, r=32, d=0.3, dist_func = 'cosine'):
    """Find candidate duplicate pairs using LSH and refine using exact cosine distance.
    
    Parameters
    ----------
    X : np.array shape [N, D]
        Data matrix.
    b : int
        Number of bands.
    r : int
        Number of rows per band.
    d : float
        Distance treshold for reporting duplicates.
    dist_func: str
        Which distance function to use, from ['cosine', 'euclidean', 'dot']
    
    Returns
    -------
    duplicates : {(ID1, ID2, d_{12}), ..., (IDX, IDY, d_{xy})}
        A set of tuples indicating the detected duplicates.
        Each tuple should have 3 elements:
            * ID of the first song
            * ID of the second song
            * The cosine distance between them
    
    n_candidates : int
        Number of detected candidate pairs.
        
    """
    n_candidates = 0
    duplicates = set()
    
    assert dist_func in ['cosine', 'euclidean', 'dot']
    
    if dist_func == 'cosine':
        dist_func = cosine_dist
        signature_func = cosine_signature
        
    elif dist_func == 'euclidean':
        dist_func = euclidean_dist
        signature_func = euclidean_signature
        
    else:
        dist_func = dot_product
        signature_func = dot_signature
    
    N, D = X.shape
    
    signature_matrix = signature_func(X, b, r)

    for band in range(b):
        
        hashes = defaultdict(list)
                
        ix = band * r
        band_matrix = signature_matrix[ix : ix+r]
        band_matrix = band_matrix == band_matrix[0, :]
        
        for i in range(N):
            hashes[hash(band_matrix[:, i].tostring())].append(i)
            
        for (h, dups) in hashes.items():
            if len(dups) <= 1:
                continue

            for i in range(len(dups)):
                n1 = dups[i]
                
                for j in range(i+1, len(dups)):
                    n2 = dups[j]
                    n_candidates += 1
                    
                    real_dist = dist_func(X[dups[i]], X[dups[j]])

                    if real_dist < d:
                        duplicates.add((dups[i], dups[j], real_dist))

    
    return duplicates, n_candidates