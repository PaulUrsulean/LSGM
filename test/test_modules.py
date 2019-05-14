import unittest
import torch

from src.model import modules
from src.model import graph_operations as go
from src.model.graph_operations import encode_onehot


class MLPModuleTests(unittest.TestCase):

    def test_mlp_module_shape(self):
        mlp = modules.MLP(10, 32, 5)

        # Output has correct dimensions
        input = torch.rand((30, 7, 10))
        out = mlp(input)
        self.assertEqual(out.size(), (30, 7, 5))

        # Assert error is thrown on wrong input dimensions
        input = torch.rand((30, 5))
        self.assertRaises(RuntimeError, mlp, input)


class RNNDecoderTests(unittest.TestCase):
    pass
    # def test_rnn_decoder_shape(self):
    #    decoder = modules.RNNDecoder(10, 2, 30)


#
#    data = torch.rand((100, 50, 5, 10))
#    graph = torch.rand((100, 5 * 4, 2)).round()
#    graph = go.encode_onehot(graph)
#    out = decoder(data, graph)


class MLPEncoderTests(unittest.TestCase):

    def __init__(self, arg):
        super(MLPEncoderTests, self).__init__(arg)

        self.N_STEPS = 4
        self.N_OBJ = 3
        self.N_FEAT = 4
        self.N_EDGE_TYPES = 2
        self.N_HIDDEN = 33

    def test_mlp_encoder_shape(self):
        encoder = modules.MLPEncoder(self.N_STEPS * self.N_FEAT, self.N_HIDDEN, self.N_EDGE_TYPES)

        input = torch.rand((100, self.N_OBJ, self.N_STEPS, self.N_FEAT))
        rel_rec, rel_send = go.gen_fully_connected(self.N_OBJ)
        out = encoder(input, rel_rec, rel_send)

        self.assertEqual(out.size(), (100, self.N_OBJ * (self.N_OBJ - 1), self.N_EDGE_TYPES))

    def test_mlp_encoder_can_learn(self):
        data = torch.rand((100, self.N_OBJ, self.N_STEPS, self.N_FEAT))

        encoder = modules.MLPEncoder(self.N_STEPS * self.N_FEAT, self.N_HIDDEN, self.N_EDGE_TYPES)

        rel_rec, rel_send = go.gen_fully_connected(self.N_OBJ)

        from torch.optim import SGD
        opt = SGD(encoder.parameters(), lr=.0001)
        opt.zero_grad()

        losses = []
        for i in range(2):
            out = encoder(data, rel_rec, rel_send)
            loss = torch.norm(out, 2)
            losses.append(loss.item())
            loss.backward()
            opt.step()

        self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
    unittest.main()
