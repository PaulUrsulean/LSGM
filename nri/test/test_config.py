import argparse
import unittest

import nri.src.config_parser as config_parser
from nri.src.config_parser import ConfigParser, generate_config
from nri.src.config import _default_config


class TestConfigParser(unittest.TestCase):

    def test_parser_changes_only_specified_values(self):
        # Only set config argument
        args = argparse.ArgumentParser()
        args.add_argument("--config", default="config.json")
        args.add_argument("--load-path", default=None)

        options = config_parser.options

        # Use no additional options
        config = ConfigParser(args,
                              args_list=['--encoder-hidden', "-234",
                                         '--batch-size', "5550123",
                                         "--store-models", "False"],
                              options=options).config

        self.assertNotEqual(_default_config, config)
        self.assertEqual(config['model']['encoder']['hidden_dim'], -234)
        self.assertEqual(config['training']['batch_size'], 5550123)
        self.assertEqual(config['logging']['store_models'], False)

    def test_generate_config(self):
        config = generate_config(encoder_hidden=-234)

        self.assertEqual(config['model']['encoder']['hidden_dim'], -234)
        self.assertNotEqual(config['model']['decoder']['hidden_dim'], -234)


if __name__ == '__main__':
    unittest.main()
