import argparse
import collections
import json
from collections import OrderedDict
from pathlib import Path
from functools import reduce
from operator import getitem


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


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
        for opt in options:
            if opt.type == bool:
                args.add_argument(opt.flag, default=None, type=str2bool)
            else:
                args.add_argument(opt.flag, default=None, type=opt.type)

        args = args.parse_args(args_list)

        # TODO Support resume
        # if args.resume:
        #    self.resume = Path(args.resume)
        #    self.cfg_fname = self.resume.parent / 'config.json'
        # else:
        #    msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        #    assert args.config is not None, msg_no_cfg
        #    self.resume = None
        #    self.cfg_fname = Path(args.config)
        self.cfg_fname = Path(args.config)

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self._config = _update_config(config, options, args)

    def __getitem__(self, name):
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
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
    CustomArgs('--store-models', type=bool, target=('logging', 'store_models')),
    CustomArgs('--gpu-id', type=int, target=('training', 'gpu_id')),
    CustomArgs('--skip-first', type=bool, target=('model', 'skip_first')),
    # CustomArgs('--prior', type= TODO,
    CustomArgs('--dynamic-graph', type=bool, target=('model', 'dynamic_graph'))
]
