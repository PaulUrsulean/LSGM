import argparse
import logging

import torch
import numpy as np

# Data loaders
from src.data_loaders.loaders import load_spring_data, load_random_data, load_weather_data

# config parser
from src.config_parser_LSTM import ConfigParser, options

# Model
from src.model.model_LSTM import RecurrentBaseline
from src.model.modules_LSTM import RNN
from src.model.utils import load_lstm_models


def run_experiment(config):

    # Random seeds
    torch.random.manual_seed(config['globals']['seed'])
    np.random.seed(config['globals']['seed'])

    logger = logging.getLogger("LSTM experiment")

    logger.debug("Creating RNN")
    rnn = create_rnn(config)

    if config['training']['load_path']:
        rnn = load_lstm_models(rnn, config)


    logger.debug("Loading data ..")
    if config['data']['name'] == 'springs':
        data_loaders = load_spring_data(batch_size=config['training']['batch_size'],
                                        suffix=config['data']['springs']['suffix'],
                                        path=config['data']['path'])

    elif config['data']['name'] == 'random':
        data_loaders = load_random_data(batch_size=config['training']['batch_size'],
                                        n_atoms=config['data']['random']['atoms'],
                                        n_examples=config['data']['random']['examples'],
                                        n_dims=config['data']['random']['dims'],
                                        n_timesteps=config['data']['random']['timesteps'])

    elif config['data']['name']=='weather':
        data_loaders = load_weather_data(batch_size=config['training']['batch_size'],
                                         n_samples=config['data']['weather']['examples'],
                                         n_nodes=config['data']['weather']['atoms'],
                                         n_timesteps=config['data']['weather']['timesteps'],
                                         features=['avg_temp', 'rainfall'],  # TODO Configurable
                                         dataset_path=config['data']['weather']['path'],
                                         force_new=config['data']['weather']['force_new'],
                                         discard=config['data']['weather']['discard'],
                                         train_valid_test_split=config['data']['weather']['splits'])

    else:
        raise NotImplementedError(config['data']['name'])

    logger.debug("Creating model...")
    model = RecurrentBaseline(rnn=rnn,
                              data_loaders=data_loaders,
                              config=config)

    logger.debug("Start training..")
    train_history = model.train()
    logger.debug(train_history[-1])

    # test loggig
    logger.debug("Running evaluation")
    test_results = model.test()
    logger.debug(test_results)


def create_rnn(config):

    dataset = config['data']['name']
    n_features = config['data'][dataset]['dims']

    rnn = RNN(n_in= n_features,                             # number of features
              n_hid=config['model']['hidden_dim'],          # number of hidden units
              n_out= n_features,                            # number of dimensions e.g. position, velocity
              n_atoms=config['data'][dataset]['atoms'],     # num of atoms in simulation
              n_layers=config['model']['num_layers'],       # number of LSTM-layers
              do_prob=config['model']['dropout'])           # Dropout-rate

    return rnn

if __name__=='__main__':
    args=argparse.ArgumentParser(description='TODO')
    args.add_argument('-c', '--config', default="config_LSTM.json", type=str,
                      help="config file path (default: None")
    args.add_argument('-l', '--load-path', default=None, type=str,
                      help='path to latest checkpoint (default: None)')


    #load config
    config =ConfigParser(args, options=options).config
    run_experiment(config)