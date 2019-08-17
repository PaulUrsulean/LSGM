import os
import pickle
import sys
from os.path import join

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .weather_loader import WeatherDataset


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


def load_weather_data_raw(batch_size, n_samples, n_nodes, n_timesteps, features, filepath):
    """
    """
    data_dict = pickle.load(open(filepath, "rb"))
    assert n_samples == len(data_dict['train_set']) \
           + len(data_dict['valid_set']) \
           + len(data_dict['test_set']), \
        "n_samples does not match sum of samples in dictionary."

    assert n_nodes == data_dict['train_set'].shape[1] \
           and n_nodes == data_dict['valid_set'].shape[1] \
           and n_nodes == data_dict['test_set'].shape[1], \
        "Number of nodes in dictionary sets do not match n_nodes"

    assert n_timesteps == data_dict['train_set'].shape[2] \
           and n_timesteps == data_dict['valid_set'].shape[2] \
           and n_timesteps == data_dict['test_set'].shape[2], \
        "Number of timestamps in dictionary sets do not match n_timesteps"

    assert len(features) == data_dict['train_set'].shape[3] \
           and len(features) == data_dict['valid_set'].shape[3] \
           and len(features) == data_dict['test_set'].shape[3], \
        "Number of features in dictionary sets do not match len(features)"

    train_data = TensorDataset(torch.FloatTensor(data_dict['train_set']))
    valid_data = TensorDataset(torch.FloatTensor(data_dict['valid_set']))
    test_data = TensorDataset(torch.FloatTensor(data_dict['test_set']))

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return dict(
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        test_loader=test_data_loader
    )


def load_weather_data(batch_size, n_samples, n_nodes, n_timesteps, features, train_valid_test_split=[80, 10, 10],
                      filename=None, dataset_path=None, force_new=False, discard=False, normalize=True):
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
        normalize(boolean, optional): Whether to center data at mean 0 and scale to stddev 1. Defaults true
    """
    # Normalization is activated when calling WeatherDataset.train_valid_test_split
    dset = WeatherDataset(n_samples, n_nodes, n_timesteps, features, filename, dataset_path, force_new, discard)
    assert len(train_valid_test_split) == 3 and sum(
        train_valid_test_split) == 100, "Invalid split given, the 3 values must sum to 100"

    # Makes actual WeatherDataset objects instead of just putting numpy arrays in the loader
    train_set, valid_set, test_set = WeatherDataset.train_valid_test_split(dset, features, train_valid_test_split,
                                                                           normalize=normalize, export=False)

    print("Split completed: {}, {}, {}".format(train_set[:].shape, valid_set[:].shape, test_set[:].shape))
    return dict(
        train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True),
        valid_loader=DataLoader(valid_set, batch_size=batch_size, shuffle=True),
        test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=True)
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


def load_data_from_config(config):
    if config['data']['name'] == 'springs':
        data_loaders = load_spring_data(batch_size=config['training']['batch_size'],
                                        suffix=config['data']['springs']['suffix'],
                                        path=join(config['data']['path'], "springs"))
    elif config['data']['name'] == 'random':
        data_loaders = load_random_data(batch_size=config['training']['batch_size'],
                                        n_atoms=config['data']['random']['atoms'],
                                        n_examples=config['data']['random']['examples'],
                                        n_dims=config['data']['random']['dims'],
                                        n_timesteps=config['data']['random']['timesteps'])
    elif config['data']['name'] == 'weather':

        filename = \
            f"{config['data']['weather']['examples']}_{config['data']['weather']['atoms']}" \
                f"_{config['data']['weather']['timesteps']}_{config['data']['weather']['dims']}" \
                f"_0_raw{config['data']['weather']['suffix']}.pickle"

        print(filename)

        features = ['avg_temp']
        if config['data']['weather']['timesteps'] == 2:
            features += ["rainfall"]

        data_loaders = load_weather_data_raw(batch_size=config['training']['batch_size'],
                                             n_samples=config['data']['weather']['examples'],
                                             n_nodes=config['data']['weather']['atoms'],
                                             n_timesteps=config['data']['weather']['timesteps'],
                                             features=features,  # TODO,
                                             filepath=join(config['data']['path'], "weather", filename))
    else:
        raise NotImplementedError(config['data']['name'])
    return data_loaders
