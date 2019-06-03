import argparse
import collections
import json
from collections import OrderedDict
from functools import reduce
from operator import getitem
from pathlib import Path

from src.config import _default_config


def read_json(fname, object_hook=OrderedDict):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=object_hook)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ConfigParser:
    # Taken from https://github.com/victoresque/pytorch-template/blob/master/parse_config.py and modified
    def __init__(self, args, args_list=None, options=''):
        # parse default and custom cli options
        options_to_args(args, options)

        args = args.parse_args(args_list)

        if args.load_path:
            self.resume = Path(args.load_path)
            self.cfg_fname = self.resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = Path(args.config)

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        if args.load_path:
            self._config = config
            self._config['training']['load_path'] = args.load_path
        else:
            self._config = _update_config(config, options, args)

    def __getitem__(self, name):
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config


def options_to_args(args, options):
    for opt in options:
        if opt.type == bool:
            args.add_argument(opt.flag, default=None, type=str2bool)
        else:
            args.add_argument(opt.flag, default=None, type=opt.type)


def _update_config(config, options, args):
    for opt in options:
        if type(args) is dict:
            value = dict.get(args, _get_opt_name(opt.flag))
        else:
            value = getattr(args, _get_opt_name(opt.flag))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flag):
    return flag.replace('--', '').replace('-', '_')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


CustomArgs = collections.namedtuple('CustomArgs', 'flag type target')
options = [
    # Globals
    CustomArgs('--seed', type=int, target=('globals', 'seed')),
    CustomArgs('--prior', type=bool, target=('globals', 'prior')),
    CustomArgs('--add-const', type=bool, target=('globals', 'add_const')),
    CustomArgs('--eps', type=float, target=('globals', 'eps')),

    # Training
    CustomArgs('--gpu-id', type=int, target=('training', 'gpu_id')),
    CustomArgs('--use-early-stopping', type=bool, target=('training', 'use_early_stopping')),
    CustomArgs('--early-stopping-metric', type=str, target=('training', 'early_stopping_metric')),
    CustomArgs('--early-stopping-patience', type=int, target=('training', 'early_stopping_patience')),
    CustomArgs('--epochs', type=int, target=('training', 'epochs')),
    CustomArgs('--batch-size', type=int, target=('training', 'batch_size')),
    CustomArgs('--lr', type=float, target=('training', 'optimizer', 'learning_rate')),  # TODO betas?
    CustomArgs('--scheduler-stepsize', type=int, target=('training', 'scheduler', 'stepsize')),
    CustomArgs('--scheduler-gamma', type=float, target=('training', 'scheduler', 'gamma')),
    CustomArgs('--grad-clip-value', type=float, target=('training', 'grad_clip_value')),

    # Spring Data
    CustomArgs('--n-timesteps', type=int, target=('data', 'timesteps')),
    CustomArgs('--dataset-name', type=str, target=('data', 'name')),
    CustomArgs('--dataset-path', type=str, target=('data', 'path')),

    # Random Data
    CustomArgs('--random-data-atoms', type=int, target=('data', 'random', 'atoms')),
    CustomArgs('--random-data-features', type=int, target=('data', 'random', 'dims')),
    CustomArgs('--random-data-timesteps', type=int, target=('data', 'random', 'timesteps')),
    CustomArgs('--random-data-examples', type=int, target=('data', 'random', 'examples')),

    # Weather Data
    CustomArgs('--weather-data-examples', type=int, target=('data', 'weather', 'examples')),
    CustomArgs('--weather-data-atoms', type=int, target=('data', 'weather', 'atoms')),
    CustomArgs('--weather-data-timesteps', type=int, target=('data', 'weather', 'timesteps')),
    CustomArgs('--weather-data-force_new', type=int, target=('data', 'weather', 'force_new')),
    CustomArgs('--weather-data-path', type=str, target=('data', 'weather', 'path')),
    CustomArgs('--weather-data-discard', type=int, target=('data', 'weather', 'discard')),

    # Loss
    CustomArgs('--loss-beta', type=float, target=('loss', 'beta')),

    # Model
    CustomArgs('--prediction-steps', type=int, target=('model', 'prediction_steps')),
    CustomArgs('--factor-graph', type=bool, target=('model', 'factor_graph')),
    CustomArgs('--skip-first', type=bool, target=('model', 'skip_first')),
    CustomArgs('--hard', type=bool, target=('model', 'hard')),
    CustomArgs('--dynamic-graph', type=bool, target=('model', 'dynamic_graph')),
    CustomArgs('--temp', type=float, target=('model', 'temp')),
    CustomArgs('--burn-in', type=bool, target=('model', 'burn_in')),
    CustomArgs('--n-edges', type=int, target=('model', 'n_edge_types')),

    # Encoder
    CustomArgs('--encoder', type=str, target=('model', 'encoder', 'model')),
    CustomArgs('--encoder-hidden', type=int, target=('model', 'encoder', 'hidden_dim')),
    CustomArgs('--encoder-dropout', type=float, target=('model', 'encoder', 'dropout')),

    # Decoder
    CustomArgs('--decoder', type=str, target=('model', 'decoder', 'model')),
    CustomArgs('--decoder-hidden', type=int, target=('model', 'decoder', 'hidden_dim')),
    CustomArgs('--decoder-dropout', type=float, target=('model', 'decoder', 'dropout')),
    CustomArgs('--prediction-var', type=float, target=('model', 'decoder', 'prediction_variance')),

    # Logging
    CustomArgs('--log-freq', type=int, target=('logging', 'log_step')),
    CustomArgs('--store-models', type=bool, target=('logging', 'store_models')),
    CustomArgs('--save-folder', type=str, target=('logging', 'log_dir')),
    # Logger config ignored

]


def generate_config(**kwargs):
    args = argparse.ArgumentParser()
    options_to_args(args, options)

    config = _update_config(_default_config.copy(),
                            options=options,
                            args=kwargs)

    return config
