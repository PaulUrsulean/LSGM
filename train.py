import argparse
import collections
import json
import logging

from src.config import generate_config
from src.config_parser import ConfigParser
from src.data.loaders import load_spring_data
from src.model import Model
from src.model.modules import MLPEncoder, RNNDecoder, CNNEncoder, MLPDecoder

#Test Change "Lukas"

def main(config):
    logger = logging.getLogger("main")

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
                      #help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default=None, type=str,
    #                  #help='path to latest checkpoint (default: None)') # TODO: Add support

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flag type target')
    options = [
        CustomArgs('--n-atoms', type=int, target=('data', 'n_atoms')),
        CustomArgs('--n-edges', type=int, target=('model', 'n_edge_types')),
        CustomArgs('--n-timesteps', type=int, target=('data', 'timesteps')),
        CustomArgs('--prediction-steps', type=int, target=('model', 'prediction_steps')),
        CustomArgs('--batch-size', type=int, target=('training', 'batch_size')),
        CustomArgs('--epochs', type=int, target=('training', 'epochs')),
        CustomArgs('--use-early-stopping', type=bool, target=('training', 'use_early_stopping')),
        CustomArgs('--patience', type=int, target=('training', 'early_stopping_patience')),
        CustomArgs('--lr', type=float, target=('training', 'optimizer', 'learning_rate')),
        CustomArgs('--lr-decay-freq', type=int, target=('training', 'scheduler', 'stepsize')),
        CustomArgs('--gamma', type=float, target=('training', 'scheduler', 'gamma')),
        CustomArgs('--temp', type=float, target=('model', 'temp')),
        CustomArgs('--var', type=float, target=('model', 'decoder', 'prediction_variance')),
        CustomArgs('--hard', type=bool, target=('model', 'hard')),
        CustomArgs('--burn-in', type=bool, target=('model', 'burn_in')),
        CustomArgs('--encoder', type=str, target=('model', 'encoder', 'model')),
        CustomArgs('--decoder', type=str, target=('model', 'decoder', 'model')),
        CustomArgs('--encoder-hidden', type=int, target=('model', 'encoder', 'hidden_dim')),
        CustomArgs('--decoder-hidden', type=int, target=('model', 'decoder', 'hidden_dim')),
        CustomArgs('--encoder-dropout', type=float, target=('model', 'encoder', 'dropout')),
        CustomArgs('--decoder-dropout', type=float, target=('model', 'decoder', 'dropout')),
        CustomArgs('--save-folder', type=str, target=('logging', 'log_dir')),
        # CustomArgs('--load-folder', type=str, default='', TODO,
        CustomArgs('--log-freq', type=int, target=('logging', 'log_step')),
        CustomArgs('--store-model', type=bool, target=('logging', 'store_models')),
        CustomArgs('--gpu-id', type=int, target=('training', 'gpu_id')),
        CustomArgs('--skip-first', type=bool, target=('model', 'skip_first')),
        # CustomArgs('--prior', type= TODO,
        CustomArgs('--dynamic-graph', type=bool, target=('model', 'dynamic_graph'))
    ]

    config = ConfigParser(args, options).config
    main(config)



#
#    config = generate_config(
#        n_edges=args.n_edges,
#        n_atoms=args.n_atoms,
#        epochs=args.epochs,
#        use_early_stopping=args.use_early_stopping,
#        early_stopping_patience=args.patience,
#        temp=args.temp,
#        gpu_id=args.gpu_id,
#
#        timesteps=args.n_timesteps,
#        prediction_steps=args.prediction_steps,
#        pred_steps=args.prediction_steps,  # Duplicate, check issue #24
#
#        hard=args.hard,
#        burn_in=args.burn_in,
#
#        log_step=args.log_freq,
#        log_dir=args.save_folder,
#        logger_config="",  # str ???
#        store_models=args.store_model,
#
#        scheduler_stepsize=args.lr_decay_freq,
#        scheduler_gamma=args.gamma,  # Decay rate of learning rate
#
#        adam_learning_rate=args.lr,  # normally 1e-3
#
#        prior=None,
#        add_const=False,
#        eps=1e-16,
#        beta=1.0,
#        prediction_variance=args.var
#    )
#
#    main(config)
#
