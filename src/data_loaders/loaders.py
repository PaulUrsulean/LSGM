import os

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader

from src.data_loaders.weather_loader import WeatherDataset


def load_spring_data(batch_size=128, suffix='', path="data/"):
    # Taken from https://github.com/ethanfetaya/NRI/blob/master/utils.py and modified

    loc_train = np.load(os.path.join(path, 'loc_train' + suffix + '.npy'))
    vel_train = np.load(os.path.join(path, 'vel_train' + suffix + '.npy'))
    edges_train = np.load(os.path.join(path, 'edges_train' + suffix + '.npy'))
    loc_valid = np.load(os.path.join(path, 'loc_valid' + suffix + '.npy'))
    vel_valid = np.load(os.path.join(path, 'vel_valid' + suffix + '.npy'))
    edges_valid = np.load(os.path.join(path, 'edges_valid' + suffix + '.npy'))
    loc_test = np.load(os.path.join(path, 'loc_test' + suffix + '.npy'))
    vel_test = np.load(os.path.join(path, 'vel_test' + suffix + '.npy'))
    edges_test = np.load(os.path.join(path, 'edges_test' + suffix + '.npy'))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return dict(
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        test_loader=test_data_loader
    )


def load_weather_data(batch_size, n_samples, n_nodes, n_timesteps, features, train_valid_test_split=[80, 10, 10],
                      filename=None, force_new=False, discard=False):
    """
    Generates the dataset with the given parameters, unless a similar dataset has been generated
        before, in which case it is by default loaded from the file.
    Args:
        n_samples(int): Number of simulations to fetch
        n_nodes(int): Number of atoms in the simulation
        n_timesteps(int): Number of timesteps in each simulation
        features(list(str)): The list of feature names to track at each time step
        filename(str): The name of the file to save to/load from. If the file already exists,
            the data is loaded from it and checked whether it matches the required parameters,
            unless the force_new parameter is set, in which case it is overwritten anyway. If
            the filename is not specified, the generator will default to a predetermined identifier
            format based on the parameters of the generated set.
        force_new(boolean, optional): Force generation of a new set of simulations,
            instead of using already existing ones from a file.
        discard(boolean, optional): Whether to discard the generated data or to save
            it, useful for debugging. Does not apply if filename is specified
    """
    dset = WeatherDataset(n_samples, n_nodes, n_timesteps, features, filename, force_new, discard)
    assert len(train_valid_test_split) == 3 and sum(
        train_valid_test_split) == 100, "Invalid split given, the 3 values must sum to 100"

    n_train = int(len(dset) * (train_valid_test_split[0] / 100))
    n_valid = int(len(dset) * (train_valid_test_split[1] / 100))

    return dict(
        train_loader=DataLoader(dset[:n_train], batch_size=batch_size),
        valid_loader=DataLoader(dset[n_train:n_train + n_valid], batch_size=batch_size),
        test_loader=DataLoader(dset[n_train + n_valid:], batch_size=batch_size)
    )


def load_random_data(batch_size, n_atoms, n_examples, n_dims, n_timesteps):
    data_loaders = dict(
        train_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_dims)),
                                     batch_size=batch_size, shuffle=True),
        valid_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_dims)),
                                     batch_size=batch_size, shuffle=True),
        test_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_dims)),
                                    batch_size=batch_size, shuffle=True)
    )
    return data_loaders
