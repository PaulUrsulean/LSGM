import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNN(nn.Module):
    """LSTM model for joint trajectory prediction."""

    def __init__(self, n_in, n_hid, n_out, n_atoms, n_layers, do_prob=0.):
        super(RNN, self).__init__()
        self.fc1_1 = nn.Linear(n_in, n_hid)
        self.fc1_2 = nn.Linear(n_hid, n_hid)
        self.rnn = nn.LSTM(n_atoms * n_hid, n_atoms * n_hid, n_layers)
        self.fc2_1 = nn.Linear(n_atoms * n_hid, n_atoms * n_hid)
        self.fc2_2 = nn.Linear(n_atoms * n_hid, n_atoms * n_out)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)

        return x.view(inputs.size(0), inputs.size(1), -1)

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_in]

        # Paper original
        x = F.relu(self.fc1_1(ins))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc1_2(x))
        x = x.view(ins.size(0), -1)
        # [num_sims, n_atoms*n_hid]

        x = x.unsqueeze(0)
        x, hidden = self.rnn(x, hidden)
        x = x[0, :, :]

        x = F.relu(self.fc2_1(x))
        x = self.fc2_2(x)
        # [num_sims, n_out*n_atoms]

        x = x.view(ins.size(0), ins.size(1), -1)
        # [num_sims, n_atoms, n_out]

        # Predict position/velocity difference
        x = x + ins

        return x, hidden

    def forward(self, inputs, prediction_steps, burn_in=False, burn_in_steps=1):
        # Input shape: [num_sims, num_things, num_timesteps, n_in]
        # #####ADDed by lukas
        inputs=inputs.view(inputs.size(0), inputs.size(1), -1)
        #new shape [num_sims, num_atoms, num_timesteps*num_dims]
        print(inputs.shape)
        outputs = []
        hidden = None

        for step in range(0, inputs.size(2) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins=inputs[:, step, :]
                    ##### commented by lukas
                    #ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]
            else:
                # Use ground truth trajectory input vs. last prediction
                if not step % prediction_steps:
                    ins=inputs[:, step, :]
                    # ####Commented by lukas
                    # ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]
            # added by lukas
            ins.float()
            output, hidden = self.step(ins, hidden)
            # Predict position/velocity difference
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)

        return outputs

