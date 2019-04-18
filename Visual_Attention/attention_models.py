import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet, GTH, get_norm
# Default concat, 1 layer, output layer
class Base_Att(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, bidirect=False,dropout=0.0):
        super(Base_Att, self).__init__()
        norm_layer = get_norm(norm)
        if(bidirect is False):
            self.nonlinear = FCNet([v_dim + q_dim, num_hid], dropout= dropout, norm= norm, act= act)
        else:
            self.nonlinear = FCNet([v_dim + 2*q_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits
