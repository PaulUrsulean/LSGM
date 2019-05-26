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


class ConfigParser:
    # Taken from https://github.com/victoresque/pytorch-template/blob/master/parse_config.py and modified
    def __init__(self, args, options=''):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(opt.flag, default=None, type=opt.type)
        args = args.parse_args()

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
