import torch.optim as optim
from torch.optim import lr_scheduler
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import time
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import os

import wandb

import gTDR.utils.DAG_GNN as utils


class Trainer:
    def __init__(self, model, args, ground_truth_G, report_log=True):
        self.model = model
        self.save, self.report_log = args.save_results, report_log
        self.args = args
        self.lr, self.c_A, self.lambda_A = args.lr, args.c_A, args.lambda_A
        self.ground_truth_G = ground_truth_G

        now = datetime.datetime.now()
        timestamp = now.isoformat()
        save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
        self.save_folder = save_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.encoder_file = os.path.join(save_folder, 'encoder.pt')
        self.decoder_file = os.path.join(save_folder, 'decoder.pt')

        log_file = os.path.join(save_folder, 'log.txt')
        self.log = open(log_file, 'w')

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=args.lr)
        elif args.optimizer == 'LBFGS':
            self.optimizer = optim.LBFGS(
                list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=args.lr)

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_decay, gamma=args.gamma)

        # Generate off-diagonal interaction graph
        off_diag = np.ones(
            [args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)

        # receiving relationship, denotes to which node a message or information is being received
        self.rel_rec = torch.DoubleTensor(
            np.array(utils.encode_onehot(np.where(off_diag)[1]), dtype=np.float64))

        # sending relationship, denotes from which node a message or information is being sent
        self.rel_send = torch.DoubleTensor(
            np.array(utils.encode_onehot(np.where(off_diag)[0]), dtype=np.float64))

        self.cs = {
            "blue": "#0072B2",
        }

    def update_optimizer(self):
        '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = self.lr / (math.log10(self.c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            self.lr = MAX_LR
        elif estimated_lr < MIN_LR:
            self.lr = MIN_LR
        else:
            self.lr = estimated_lr

        # set LR
        for parame_group in self.optimizer.param_groups:
            parame_group['lr'] = self.lr

    def train_one_epoch(self, train_loader, rel_rec, rel_send,
                        epoch, best_loss, ground_truth_G):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        self.model.encoder.train()
        self.model.decoder.train()
        self.scheduler.step()

        # update optimizer
        self.update_optimizer()

        for batch_idx, (data, relations) in enumerate(train_loader):

            if self.args.use_cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(
                data).double(), Variable(relations).double()

            # reshape data
            relations = relations.unsqueeze(2)

            self.optimizer.zero_grad()

            logits, self.origin_A, myA, output = self.model.forward(
                data, rel_rec, rel_send)

            if torch.sum(output != output):
                print('nan error\n')

            target = data
            preds = output
            variance = 0.

            # reconstruction accuracy loss
            loss_nll = utils.nll_gaussian(preds, target, variance)
            # KL loss
            loss_kl = utils.kl_gaussian_sem(logits)
            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            sparse_loss = self.args.tau_A * torch.sum(torch.abs(self.origin_A))

            # compute h(A)
            h_A = utils.compute_h_A(
                self.origin_A, self.args.data_variable_size)
            loss += self.lambda_A * h_A + 0.5 * self.c_A * h_A * h_A + 100. * \
                torch.trace(self.origin_A*self.origin_A) + \
                sparse_loss  # +  0.01 * torch.sum(variance * variance)

            loss.backward()
            self.optimizer.step()

            myA.data = utils.stau(myA.data, self.args.tau_A * self.lr)

            if torch.sum(self.origin_A != self.origin_A):
                print('nan error\n')

            # compute metrics
            graph = self.origin_A.data.cpu().clone().numpy()
            graph[np.abs(graph) < self.args.graph_threshold] = 0

            fdr, tpr, fpr, shd, nnz = utils.count_accuracy(
                ground_truth_G, nx.DiGraph(graph))

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            shd_trian.append(shd)

        # log metrics
        if self.use_wandb:
            wandb.log({
                "nll_train": np.mean(nll_train),
                "kl_train": np.mean(kl_train),
                "mse_train": np.mean(mse_train),
                "shd_train": np.mean(shd_trian),
                "ELBO_loss": np.mean(kl_train) + np.mean(nll_train)
            })

        # best so far
        if np.mean(nll_train) < best_loss:
            if self.save:
                self.save_best_checkpoint()
                # print('Best model so far, saving...')

            if self.report_log:
                print('Epoch: {:04d}'.format(epoch),
                      'nll_train: {:.10f}'.format(np.mean(nll_train)),
                      'kl_train: {:.10f}'.format(np.mean(kl_train)),
                      'ELBO_loss: {:.10f}'.format(
                          np.mean(kl_train) + np.mean(nll_train)),
                      'mse_train: {:.10f}'.format(np.mean(mse_train)),
                      'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
                      'time: {:.4f}s'.format(time.time() - t), file=self.log)
                if self.log is not None:
                    self.log.flush()

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, self.origin_A

    def train(self, train_loader, use_wandb=False, parameter_search=False):
        self.use_wandb = use_wandb
        if callable(parameter_search):
            self.use_wandb = True
            parameter_search(self)
        best_ELBO_loss = np.inf
        best_NLL_loss = np.inf
        best_MSE_loss = np.inf
        self.all_ELBO_loss = []  # for visualization
        best_epoch = 0
        best_ELBO_graph = []
        best_NLL_graph = []
        best_MSE_graph = []

        # optimizer step on hyperparameters
        h_A_new = torch.tensor(1.)
        h_A_old = np.inf

        if self.args.use_cuda:
            self.rel_rec = self.rel_rec.cuda()
            self.rel_send = self.rel_send.cuda()

        # allows the computation of gradients
        rel_rec = Variable(self.rel_rec)
        rel_send = Variable(self.rel_send)

        for step_k in range(self.args.k_max_iter):
            while self.args.c_A < 1e+20:
                for epoch in range(self.args.epochs):
                    ELBO_loss, NLL_loss, MSE_loss, graph, self.origin_A = self.train_one_epoch(train_loader, rel_rec, rel_send, epoch,
                                                                                               best_ELBO_loss, self.ground_truth_G)
                    self.all_ELBO_loss.append(ELBO_loss)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                print(
                    "Best epoch so far for current step_k {:04d}: ".format(step_k))
                print("\tBest ELBO_loss: {:.10f}, Best NLL_loss: {:.10f}, Best MSE_loss: {:.10f}".format(
                    best_ELBO_loss, best_NLL_loss, best_MSE_loss))

                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = self.origin_A.data.clone()
                h_A_new = utils.compute_h_A(
                    A_new, self.args.data_variable_size)
                if h_A_new.item() > 0.25 * h_A_old:
                    self.c_A *= 10
                else:
                    break
            print()
            h_A_old = h_A_new.item()
            self.lambda_A += self.c_A * h_A_new.item()

            if h_A_new.item() <= float(self.args.h_tol):
                break

        print("Optimization Finished!")
        if callable(parameter_search):
            self.run.finish()

        if self.report_log:
            print("Best Epoch: {:04d}".format(best_epoch), file=self.log)
            print("Best ELBO_loss: {:.10f}, Best NLL_loss: {:.10f}, Best MSE_loss: {:.10f}".format(
                best_ELBO_loss, best_NLL_loss, best_MSE_loss), file=self.log)
            self.log.flush()

        return best_ELBO_graph, best_NLL_graph, best_MSE_graph

    def save_best_checkpoint(self):
        torch.save(self.model.encoder.state_dict(), self.encoder_file)
        torch.save(self.model.decoder.state_dict(), self.decoder_file)

    def load_best_checkpoint(self):
        self.model.encoder.load_state_dict(torch.load(self.encoder_file))
        self.model.decoder.load_state_dict(torch.load(self.decoder_file))

    def load_checkpoint(self, encoder_path, decoder_path):
        self.model.encoder.load_state_dict(torch.load(encoder_path))
        self.model.decoder.load_state_dict(torch.load(decoder_path))
        print("~~~ Loaded given checkpoint ~~~")

    def visualize_train(self):
        """At training time"""
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot(self.all_ELBO_loss, color=self.cs["blue"],
                alpha=1.0, lw=4.0, label='train ELBO loss')  # todo

        ax.set_xlabel('Iteration', labelpad=5)
        ax.set_ylabel(f'Log(ELBO loss)', labelpad=5)
        plt.yscale('log')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

        if self.save:
            plt.savefig(f"{self.save_folder}/train_loss.png", transparent=True, bbox_inches='tight', pad_inches=0)
            
