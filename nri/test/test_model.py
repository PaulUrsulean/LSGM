import unittest

from nri.src.config_parser import generate_config
from nri.src import load_random_data
from nri.src import Model, MLPDecoder
from nri.src import MLPEncoder, RNNDecoder


class MyTestCase(unittest.TestCase):

    def test_run_epoch(self):
        n_feat = 1
        n_edges = 1
        n_epochs = 2
        n_timesteps = 2
        n_hid = 1

        config = generate_config(n_timesteps=n_timesteps,
                                 n_edges=n_edges,
                                 prediction_steps=1,
                                 epochs=n_epochs,
                                 use_early_stopping=False,
                                 gpu_id=None,
                                 log_dir="/tmp/1",
                                 dataset_name='random',
                                 random_data_atoms=2,
                                 random_data_features=n_feat,
                                 random_data_timesteps=n_timesteps * 2,
                                 random_data_examples=1
                                 )

        data_loaders = load_random_data(batch_size=1,
                                        n_atoms=config['data']['random']['atoms'],
                                        n_examples=config['data']['random']['examples'],
                                        n_dims=config['data']['random']['dims'],
                                        n_timesteps=config['data']['random']['timesteps'])

        encoder = MLPEncoder(n_timesteps * n_feat, n_hid, n_edges)
        decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        trainer = Model(encoder=encoder,
                        decoder=decoder,
                        data_loaders=data_loaders,
                        config=config)

        trainer.test()
        trainer.test()
        # No errors thrown

    def test_overfit_epoch(self):
        n_feat = 1
        n_edges = 1
        n_epochs = 500
        n_timesteps = 20
        n_hid = 200

        config = generate_config(n_timesteps=n_timesteps,
                                 n_edges=n_edges,
                                 seed=42,
                                 prediction_steps=2,
                                 epochs=n_epochs,
                                 use_early_stopping=False,
                                 gpu_id=None,
                                 log_dir="/tmp/1",
                                 dataset_name='random',
                                 random_data_atoms=2,
                                 random_data_features=n_feat,
                                 random_data_timesteps=n_timesteps * 2,
                                 random_data_examples=1
                                 )

        data_loaders = load_random_data(batch_size=128,
                                        n_atoms=config['data']['random']['atoms'],
                                        n_examples=config['data']['random']['examples'],
                                        n_dims=config['data']['random']['dims'],
                                        n_timesteps=config['data']['random']['timesteps'])

        encoder = MLPEncoder(n_timesteps * n_feat, n_hid, n_edges)
        decoder = MLPDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid, msg_hid=n_hid, msg_out=n_hid)
        #decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        trainer = Model(encoder=encoder,
                        decoder=decoder,
                        data_loaders=data_loaders,
                        config=config)

        history = trainer.train()
        last_log = history[-1]
        test_out = trainer.test()

        # Assert training loss smaller than validation loss
        self.assertLess(last_log['loss'], last_log['val_loss'])
        # Assert validation mse loss increased in second half of training
        self.assertGreater(last_log['val_mse_loss'], history[n_epochs // 2]['val_mse_loss'])


if __name__ == '__main__':
    unittest.main()
