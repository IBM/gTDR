import torch.nn as nn
import torch.nn.functional as F
from gTDR.models.GANF.NF import MAF, RealNVP
import torch

class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D

        h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class GANF(nn.Module):

    def __init__ (self, args, model="MAF"):
        super(GANF, self).__init__()

        self.rnn = nn.LSTM(input_size=args.input_size,hidden_size=args.hidden_size,batch_first=True, dropout=args.dropout)
        self.gcn = GNN(input_size=args.hidden_size, hidden_size=args.hidden_size)
        if model=="MAF":
            self.nf = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, cond_label_size=args.hidden_size, batch_norm=args.batch_norm,activation='tanh')
        else:
            self.nf = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, cond_label_size=args.hidden_size, batch_norm=args.batch_norm)

    def forward(self, x, A):

        return self.test(x, A).mean()

    def test(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)

        return log_prob
    
    def locate(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],full_shape[1],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=2)

        return log_prob