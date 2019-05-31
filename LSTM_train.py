import argparse
import logging
import torch
import numpy as np

from src.data_loaders.loaders import load_spring_data, load_random_data

from LSTM_Baseline.src.config_parser_LSTM import ConfigParser, options
from LSTM_Baseline.src.model.model_LSTM import RecurrentBaseline
from LSTM_Baseline.src.model.modules_LSTM import RNN


def run_experiment(config):
    print('Accessing run_experiment')

    # Random seeds
    torch.random.manual_seed(config['globals']['seed'])
    np.random.seed(config['globals']['seed'])

    logger = logging.getLogger("LSTM experiment")

    logger.debug("Creating RNN")
    rnn = create_rnn(config)

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
    else:
        raise NotImplementedError(config['data']['name'])
    """   
    To-Do: Add load_weather_data 
    elif config['data']['weather'] == 'weather':
    #    data_loaders = load_weather_data(batch_size=config['training']['batch_size'],
    ##                                   n_atoms=config['data']['random']['atoms'],
    #                                  n_examples=config['data']['random']['examples'],
    #                                 n_dims=config['data']['random']['dims'],
    #          config_LSTM                      n_timesteps=config['data']['random']['timesteps'])
    """
    logger.debug("Creating model...")
    model = RecurrentBaseline(rnn=rnn,
                              data_loaders=data_loaders,
                              config=config)

    logger.debug("start training..")
    train_history=model.train()


def create_rnn(config):


    dataset = config['data']['name']
    n_features = config['data'][dataset]['dims']

    rnn = RNN(n_in= n_features,
              n_hid=config['model']['hidden_dim'],
              n_out= n_features,  #number of dimensions e.g. position, velocity
              n_atoms=config['data'][dataset]['atoms'], #num of atoms in simulation
              n_layers=config['model']['num_layers'],
              do_prob=config['model']['dropout'])

    return rnn

if __name__=='__main__':
    args=argparse.ArgumentParser(description='TODO')
    args.add_argument('-c', '--config', default=None, type=str)

    #load config
    config =ConfigParser(args, options=options).config

    print("debug")
    run_experiment(config)