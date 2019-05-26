import argparse
import logging

from src.config_parser import ConfigParser, options
from src.data_loaders.loaders import load_spring_data, load_random_data
from src.model import Model
from src.model.modules import MLPEncoder, RNNDecoder, CNNEncoder, MLPDecoder


def run_experiment(config):
    logger = logging.getLogger("experiment")

    logger.debug("Creating encoder and decoder")
    encoder = create_encoder(config)
    decoder = create_decoder(config)

    # TODO Load model if configured

    # TODO Add other data loaders
    logger.debug("Loading data...")
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

    logger.debug("Creating model...")
    model = Model(encoder=encoder,
                  decoder=decoder,
                  data_loaders=data_loaders,
                  config=config)

    logger.debug("Starting training")
    train_history = model.train()
    logger.debug(train_history[-1])

    logger.debug("Running evaluation")
    test_results = model.test()
    logger.debug(test_results)


def create_decoder(config):
    dataset = config['data']['name']
    n_features = config['data'][dataset]['dims']
    if config['model']['decoder']['model'] == 'mlp':
        decoder = MLPDecoder(n_in_node=n_features,
                             edge_types=config['model']['n_edge_types'],
                             msg_hid=config['model']['decoder']['hidden_dim'],
                             msg_out=config['model']['decoder']['hidden_dim'],
                             n_hid=config['model']['decoder']['hidden_dim'],
                             do_prob=config['model']['decoder']['dropout'],
                             skip_first=config['model']['skip_first'])
    elif config['model']['decoder']['model'] == 'rnn':
        decoder = RNNDecoder(n_in_node=n_features,
                             edge_types=config['model']['n_edge_types'],
                             n_hid=config['model']['decoder']['hidden_dim'],
                             do_prob=config['model']['decoder']['dropout'],
                             skip_first=config['model']['skip_first'])
    return decoder


def create_encoder(config):
    dataset = config['data']['name']
    n_features = config['data'][dataset]['dims']
    if config['model']['encoder']['model'] == 'mlp':
        encoder = MLPEncoder(n_in=config['data']['timesteps'] * n_features,
                             n_hid=config['model']['encoder']['hidden_dim'],
                             n_out=config['model']['n_edge_types'],
                             do_prob=config['model']['encoder']['dropout'],
                             factor=config['model']['factor_graph'])
    elif config['model']['encoder']['model'] == 'cnn':
        encoder = CNNEncoder(n_in=n_features,
                             n_hid=config['model']['encoder']['hidden_dim'],
                             n_out=config['model']['n_edge_types'],
                             do_prob=config['model']['encoder']['dropout'],
                             factor=config['model']['factor_graph'])
    return encoder


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='TODO')  # TODO Name
    args.add_argument('-c', '--config', default=None, type=str)
    # help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default=None, type=str,
    #                  #help='path to latest checkpoint (default: None)') # TODO: Add support

    config = ConfigParser(args, options=options).config
    run_experiment(config)