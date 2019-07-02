from collections import defaultdict
import torch

import numpy as np


def cosine_dist(v1, v2, use_tensors):
    return 1 - (v1 @ v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_dist(v1, v2, use_tensors):
    return np.linalg.norm(v1 - v2)

def dot_product(v1, v2, use_tensors):
    return torch.dot(v1, v2) if use_tensors else v1 @ v2

def cosine_signature(X, b, r, use_tensors):
    N, D = X.shape
    random_projections = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D), size=b*r)
    
    inside = torch.mm(torch.tensor(random_projections, dtype=torch.float), X.t()) if use_tensors else random_projections @ X.T
    return (inside >= 0) * 2 - 1

def euclidean_signature(X, b, r, use_tensors, w=0.25):
    N, D = X.shape
    random_projections = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D), size=b*r)
    bias = np.random.uniform(0, w, size=b*r)
    
    sketch = (torch.mm(torch.tensor(random_projections, dtype=torch.float), X.t()) + torch.tensor(bias[np.newaxis].T, dtype=torch.float)) / torch.tensor(w, dtype=torch.float) if use_tensors else (random_projections @ X.T + bias[np.newaxis].T) / w
    
    return torch.floor(sketch) if use_tensors else np.floor(sketch)

def dot_signature(X, b, r, use_tensors):
    N, D = X.shape
    normalized = X/np.linalg.norm(X, axis=1).max()
    augment_col = np.sqrt(1 - np.linalg.norm(normalized, axis=1))[np.newaxis].T
    augmented = np.concatenate((normalized, augment_col), axis=1)
    random_vectors = np.random.multivariate_normal(mean=np.zeros(D+1), cov=np.eye(D+1), size=b*r)
    
    sketch = (random_vectors @ augmented.T >= 0) * 2 - 1
    
    return torch.tensor(sketch, dtype=torch.float) if use_tensors else sketch

def LSH(X, b=8, r=32, d=0.3, dist_func = 'cosine', debug=False, use_tensors=True):
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
    d = np.asscalar(d) if type(d) is np.ndarray else d
    
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
    
    signature_matrix = signature_func(X, b, r, use_tensors)

    for band in range(b):
        
        hashes = defaultdict(list)
                
        ix = band * r
        band_matrix = signature_matrix[ix : ix+r]
        band_matrix = band_matrix == band_matrix[0, :]
        
        for i in range(N):
            if use_tensors:
                hashes[hash(band_matrix[:, i].numpy().tostring())].append(i)
            else:
                hashes[hash(band_matrix[:, i].tostring())].append(i)
            
        for (h, dups) in hashes.items():
            if len(dups) <= 1:
                continue

            for i in range(len(dups)):
                n1 = dups[i]
                
                for j in range(i+1, len(dups)):
                    n2 = dups[j]
                    n_candidates += 1
                    
                    real_dist = dist_func(X[dups[i]], X[dups[j]], use_tensors)

                    # For dot product we aim to find maximizing pairs
                    if dist_func is not dot_product:
                        if real_dist < d:
                            duplicates.add((dups[i], dups[j], real_dist))
                    else:
                        if real_dist > d:
                            duplicates.add((dups[i], dups[j], real_dist))

    
    return duplicates, n_candidates if not debug else hashes