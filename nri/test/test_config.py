import unittest
import argparse

import config_parser as config_parser
from config_parser import ConfigParser, read_json, generate_config
from nri.src import _default_config


class TestConfigParser(unittest.TestCase):

    def test_parser_keeps_defaults(self):
        # Only set config argument
        args = argparse.ArgumentParser()
        args.add_argument("--config", default=None)
        args.add_argument("--load-path", default=None)

        # Use no additional options
        config = ConfigParser(args, options="").config

        default = _default_config

        self.assertEqual(default, config)

    def test_parser_changes_only_specified_values(self):
        # Only set config argument
        args = argparse.ArgumentParser()
        args.add_argument("--config", default=None)
        args.add_argument("--load-path", default=None)

        options = config_parser.options

        # Use no additional options
        config = ConfigParser(args,
                              args_list=['--encoder-hidden', "-234",
                                         '--batch-size', "5550123",
                                         "--store-models", "False"],
                              options=options).config

        # Load file directly
        default = read_json(default_file)

        self.assertNotEqual(default, config)
        self.assertEqual(config['model']['encoder']['hidden_dim'], -234)
        self.assertEqual(config['training']['batch_size'], 5550123)
        self.assertEqual(config['logging']['store_models'], False)

    def test_generate_config(self):
        config = generate_config(encoder_hidden=-234)

        self.assertEqual(config['model']['encoder']['hidden_dim'], -234)
        self.assertNotEqual(config['model']['decoder']['hidden_dim'], -234)


if __name__ == '__main__':
    unittest.main()
