import unittest
import torch

from src.model import modules
from src.model import graph_operations as go


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
    #def test_rnn_decoder_shape(self):
    #    decoder = modules.RNNDecoder(10, 2, 30)
#
    #    data = torch.rand((100, 50, 5, 10))
    #    graph = torch.rand((100, 5 * 4, 2)).round()
    #    graph = go.encode_onehot(graph)
    #    out = decoder(data, graph)


class MLPEncoderTests(unittest.TestCase):

    def __init__(self, arg):
        super(MLPEncoderTests, self).__init__(arg)

        self.N_STEPS = 30
        self.N_OBJ = 5
        self.N_FEAT = 30
        self.N_EDGE_TYPES = 10
        self.N_HIDDEN = 33

    def test_mlp_encoder_shape(self):
        encoder = modules.MLPEncoder(self.N_STEPS * self.N_FEAT, self.N_HIDDEN, self.N_EDGE_TYPES)

        input = torch.rand((100, self.N_OBJ, self.N_STEPS, self.N_FEAT))
        adj_send = torch.rand(size=(self.N_OBJ, self.N_OBJ))
        adj_rec = adj_send
        out = encoder(input, adj_rec, adj_send)

        self.assertEqual(out.size(), (100, self.N_OBJ, self.N_EDGE_TYPES))

    def test_mlp_encoder_can_learn(self):
        data = torch.rand((100, self.N_OBJ, self.N_STEPS, self.N_FEAT))

        adj_send = torch.rand(size=(self.N_OBJ, self.N_OBJ))
        adj_rec = adj_send

        encoder = modules.MLPEncoder(self.N_STEPS * self.N_FEAT, self.N_HIDDEN, self.N_EDGE_TYPES)

        from torch.optim import SGD
        opt = SGD(encoder.parameters(), lr=.0001)
        opt.zero_grad()

        losses = []
        for i in range(10):
            out = encoder(data, adj_rec, adj_send)
            loss = torch.norm(out, 2)
            losses.append(loss.item())
            loss.backward()
            opt.step()

        self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
    unittest.main()
