import torch
import time
import datetime
import os
import math
import numpy as np
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch.autograd import Variable

import gTDR.utils.DAG_GNN as dag_gnn_utils

_EPS = 1e-10

class DAG_GNN(nn.Module):
    def __init__(self, args):
        super(DAG_GNN, self).__init__()
        self.data_variable_size = args.data_variable_size
        self.x_dims = args.x_dims
        self.z_dims = args.z_dims
        self.use_cuda = args.use_cuda

        self.adj_A = np.zeros((args.data_variable_size, args.data_variable_size))
        
        save_folder = args.save_folder
        self.exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        self.save_folder_dir = '{}/exp{}/'.format(save_folder, timestamp)

        os.makedirs(self.save_folder_dir)
        self.encoder_file = os.path.join(self.save_folder_dir, 'encoder.pt')
        self.decoder_file = os.path.join(self.save_folder_dir, 'decoder.pt')

        self.log_file = os.path.join(self.save_folder_dir, 'log.txt')
        self.log = open(self.log_file, 'w')

        if args.encoder == 'mlp':
            self.encoder = MLPEncoder(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
                                      args.z_dims, self.adj_A,
                                      batch_size=args.batch_size,
                                      do_prob=args.encoder_dropout).double()
        elif args.encoder == 'sem':
            self.encoder = SEMEncoder(args.data_variable_size * args.x_dims, args.encoder_hidden,
                                      args.z_dims, self.adj_A,
                                      batch_size=args.batch_size,
                                      do_prob=args.encoder_dropout).double()

        if args.decoder == 'mlp':
            self.decoder = MLPDecoder(args.data_variable_size * args.x_dims,
                                      args.z_dims, args.x_dims, self.encoder,
                                      data_variable_size=args.data_variable_size,
                                      batch_size=args.batch_size,
                                      n_hid=args.decoder_hidden,
                                      do_prob=args.decoder_dropout).double()
        elif args.decoder == 'sem':
            self.decoder = SEMDecoder(args.data_variable_size * args.x_dims,
                                      args.z_dims, 2, self.encoder,
                                      data_variable_size=args.data_variable_size,
                                      batch_size=args.batch_size,
                                      n_hid=args.decoder_hidden,
                                      do_prob=args.decoder_dropout).double()
        
        if self.use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, data, rel_rec, rel_send):
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = self.encoder(data, rel_rec, rel_send)
        dec_x, output, adj_A_tilt_decoder = self.decoder(data, logits, self.data_variable_size * self.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)
        return logits, origin_A, myA, output
    

class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        # self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs, rel_rec, rel_send):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = dag_gnn_utils.preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class SEMEncoder(nn.Module):
    """SEM encoder module."""
    def __init__(self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(SEMEncoder, self).__init__()
        # self.factor = factor
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad = True))
        self.dropout_prob = do_prob
        self.batch_size = batch_size

    def init_weights(self):
        nn.init.xavier_normal(self.adj_A.data)

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        adj_A1 = torch.sinh(3.*self.adj_A)
        adj_A = dag_gnn_utils.preprocess_adj_new((adj_A1))
        adj_A_inv = dag_gnn_utils.preprocess_adj_new1((adj_A1))

        meanF = torch.matmul(adj_A_inv, torch.mean(torch.matmul(adj_A, inputs), 0))
        logits = torch.matmul(adj_A, inputs-meanF)

        return inputs-meanF, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):
        adj_A_new1 = dag_gnn_utils.preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt

class SEMDecoder(nn.Module):
    """SEM decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(SEMDecoder, self).__init__()

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):
        adj_A_new1 = dag_gnn_utils.preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa)
        out = mat_z

        return mat_z, out-Wa, adj_A_tilt
