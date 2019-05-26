import os
import unittest
import argparse
from pathlib import Path

import src.config_parser as config_parser
from src.config_parser import ConfigParser, read_json
from src.config import generate_config, _default_config

default_file = Path(os.path.join("res", "config.json"))


class TestConfigParser(unittest.TestCase):

    def test_parser_keeps_defaults(self):
        # Only set config argument
        args = argparse.ArgumentParser()
        args.add_argument("--config", default=default_file)

        # Use no additional options
        config = ConfigParser(args, options="").config

        # Load file directly
        default = read_json(default_file)

        self.assertEqual(default, config)

    def test_parser_changes_only_specified_values(self):
        # Only set config argument
        args = argparse.ArgumentParser()
        args.add_argument("--config", default=default_file)

        options = config_parser.options

        # Use no additional options
        config = ConfigParser(args,
                              args_list=['--encoder-hidden', "-234",
                                         '--batch-size', "5550123",
                                         "--store-models", "False"], # TODO
                              options=options).config

        # Load file directly
        default = read_json(default_file)

        self.assertNotEqual(default, config)
        self.assertEqual(config['model']['encoder']['hidden_dim'], -234)
        self.assertEqual(config['training']['batch_size'], 5550123)
        self.assertEqual(config['logging']['store_models'], False)

    def test_parse_arguments(self):
        pass


if __name__ == '__main__':
    unittest.main()
