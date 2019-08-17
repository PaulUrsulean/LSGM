import argparse
import logging

import numpy as np
import torch.nn

from nri.src import create_decoder_from_config, create_encoder_from_config, \
    load_data_from_config
from nri.src import load_weights_for_model
from nri.src.config_parser import ConfigParser, options
from nri.src.model import Model


def run_experiment(config: dict):
    """
    Runs experiment with details specified in config file.
    If a logger is specified, all results with config will `be stored in a new folder.
    :param config: Dictionary containing all fields. See 'config.json' for an example.
    :return: None
    """
    # Set Random Seeds
    torch.random.manual_seed(config['globals']['seed'])
    np.random.seed(config['globals']['seed'])

    logger = logging.getLogger("experiment")

    logger.debug("Creating encoder and decoder")
    encoder = create_encoder_from_config(config)
    decoder = create_decoder_from_config(config)

    if config['training']['load_path']:
        encoder, decoder = load_weights_for_model(encoder, decoder, config)

    logger.debug("Loading data...")
    data_loaders = load_data_from_config(config)

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


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Run NRI experiment with details specified in a config file.")
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='Path to "config.json". Leave empty to use example file in same directory.')
    args.add_argument('-l', '--load-path', default=None, type=str,
                      help='Path to existing experiment folder to automatically load latest checkpoint (default: None)')

    experiment_config = ConfigParser(args, options=options).config
    run_experiment(experiment_config)
