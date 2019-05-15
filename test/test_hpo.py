import unittest

from src.hpo import *

class TestHPO(unittest.TestCase):




    def test_something(self):

        train_config = None
        parameters = []

        hpo = HPO(train_config,
                  parameters)
        self.assertEqual(hpo.n_gpus, 1)



if __name__ == '__main__':
    unittest.main()
