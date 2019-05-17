import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model.utils import node2edge, edge2node, gen_fully_connected


class MLP(nn.Module):
    """
    Small Multilayer Perceptron Block mostly used in Encoder.
    Uses ELU for activation function and additionally uses dropout and batchnormalization for regularization.
    """

    def __init__(self, input_size, hidden_size, output_size, keep_prob=1.):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout = nn.Dropout(1.0 - keep_prob)
        self.batchnorm = nn.BatchNorm1d(output_size)

        self.linear_input_hidden = nn.Linear(input_size, hidden_size)
        self.linear_hidden_out = nn.Linear(hidden_size, output_size)

    def shape_invariant_batchnorm(self, input):
        """
        As we use a 1d-batchnorm but want to allow for arbitarary tensor shapes (e.g. running for multiple timesteps and
        objects in one pass), we have to scale the tensor before passing it into the operation
        :return: Tensor with same shape but batchnorm applied
        """
        x = input.view(input.size(0) * input.size(1),
                       -1)  # [N_TIMESTEPS, N_OBJ, N_FEAT] -> [N_TIMESTEPS * N_OBJ, N_FEAT]
        x = self.batchnorm(x)
        x = x.view(input.size(0), input.size(1), -1)  # -> [N_TIMESTEPS, N_OBJ, N_FEAT]
        return x

    def forward(self, input):
        x = self.linear_input_hidden(input)
        x = F.elu(x)
        x = self.linear_hidden_out(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.shape_invariant_batchnorm(x)
        return x


class MLPEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, keep_prob=1., n_layers=2):
        super(MLPEncoder, self).__init__()

        self.mlp1 = MLP(input_size, hidden_size, hidden_size)
        self.mlp2 = MLP(hidden_size * 2, hidden_size, hidden_size)
        self.mlp3 = MLP(hidden_size, hidden_size, hidden_size)
        self.mlp4 = MLP(hidden_size * 3, hidden_size, hidden_size)
        self.mlp_out = nn.Linear(hidden_size, output_size)

    def forward(self, input, adj_rec, adj_send):
        """
        Implementation from (Kipf et al, 2018)
        Currently uses factor-version (see paper for details)
        :param input: Tensor of shape TODO
        :return:
        """

        # Transform to column-stacking o ftimesteps and features
        x = input.view(input.size(0), input.size(1), -1)

        x = self.mlp1(x)
        x = node2edge(x, adj_rec, adj_send)
        x = self.mlp2(x)
        x_skip = x

        x = edge2node(x, adj_rec, adj_send)
        x = self.mlp3(x)
        x = node2edge(x, adj_rec, adj_send)
        x = torch.cat((x, x_skip), dim=2)  # Skip connection
        x = self.mlp4(x)
        return self.mlp_out(x)


class RNNDecoder(nn.Module):
    # Taken from https://github.com/ethanfetaya/NRI with adaptions from us
    """Recurrent decoder module."""

    def __init__(self, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        # print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))
        if inputs.is_cuda:
            all_msgs = all_msgs.to(inputs.device)

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec=None, rel_send=None, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        if rel_send is None or rel_rec is None:
            rel_rec, rel_send = gen_fully_connected(inputs.size(2), inputs.device)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = Variable(torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            hidden = hidden.to(inputs.device)

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = F.gumbel_softmax(logits, tau=temp, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)
        return preds.transpose(1, 2).contiguous()
