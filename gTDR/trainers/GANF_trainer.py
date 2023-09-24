import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_value_
import os
import matplotlib.pyplot as plt
import wandb


class Trainer:
    def __init__(self, args, model, n_sensor, train_loader, val_loader, test_loader, has_label=True):
        self.args = args
        self.has_label= has_label
        self.model = model
        if args.use_cuda:
            self.model = self.model.to(args.device)
        self.n_sensor = n_sensor
        if args.graph_dir != None:
            init = torch.load(args.graph_dir).to(args.device).abs()
            print("Load graph from", args.graph_dir)
        else:
            from torch.nn.init import xavier_uniform_
            init = torch.zeros([n_sensor, n_sensor])
            init = xavier_uniform_(init).abs()
            init = init.fill_diagonal_(0.0)
        self.A = torch.tensor(init, requires_grad=True, device=args.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.roc_val_best = 0
        self.best_loss = float('inf') 
        self.h_A_old = np.inf
        self.epoch = 0
        self.all_roc_train = []
        self.all_roc_val = []
        self.all_roc_test = []
        self.cs = {
            "orange": "#E69F00",
            "blue": "#0072B2",
                    "red": "#D55E00",
        }

        save_path = os.path.join(args.output_dir, args.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.args.save_path = save_path

    def train(self, use_wandb=False, parameter_search=False):
        self.use_wandb = use_wandb
        if callable(parameter_search):
            self.use_wandb = True
            parameter_search(self)
        rho = self.args.rho
        alpha = self.args.alpha
        h_tol = self.args.h_tol
        rho_max = self.args.rho_max

        for _ in range(self.args.max_iter):
            while rho < rho_max:
                optimizer = torch.optim.Adam([
                    {'params': self.model.parameters(
                    ), 'weight_decay': self.args.weight_decay},
                    {'params': [self.A]}], lr=self.args.lr, weight_decay=0.0)

                for _ in range(self.args.n_epochs):
                    self.run_epoch(optimizer, rho, alpha)

                if self.h.item() > 0.5 * self.h_A_old:
                    rho *= 10
                else:
                    break

            self.h_A_old = self.h.item()
            alpha += rho * self.h.item()

            if self.h_A_old <= h_tol or rho >= rho_max:
                break

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'weight_decay': self.args.weight_decay},
            {'params': [self.A]}], lr=self.args.lr, weight_decay=0.0)

        for _ in range(self.args.additional_iter):
            self.run_epoch(optimizer, rho, alpha, final_epochs=True)

        if callable(parameter_search):
            self.run.finish()

    def run_epoch(self, optimizer, rho, alpha, final_epochs=False):
        loss_train = []
        self.epoch += 1
        self.model.train()

        for x in self.train_loader:
            x = x.to(self.args.device)

            optimizer.zero_grad()
            A_hat = torch.divide(self.A.T, self.A.sum(dim=1).detach()).T ##
            loss = -self.model(x, A_hat)
            h = torch.trace(torch.matrix_exp(A_hat * A_hat)) - self.n_sensor
            # loss = -self.model(x, self.A)
            # h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.n_sensor
            total_loss = loss + 0.5 * rho * h * h + alpha * h

            total_loss.backward()
            clip_grad_value_(self.model.parameters(), 1)
            optimizer.step()
            loss_train.append(loss.item())
            self.A.data.copy_(torch.clamp(self.A.data, min=0, max=1))
        self.h = h

        self.eval_model(loss_train, final_epochs)

    def eval_model(self, loss_train, final_epochs):
        self.model.eval()
        loss_val, loss_test = [], []
        for loader, loss_list in [(self.val_loader, loss_val), (self.test_loader, loss_test)]:
            with torch.no_grad():
                for x in loader:
                    x = x.to(self.args.device)
                    loss = -self.model.test(x, self.A.data).cpu().numpy()
                    loss_list.append(loss)

        loss_val, loss_test = map(np.concatenate, (loss_val, loss_test))

        loss_val, loss_test = map(np.nan_to_num, (loss_val, loss_test))
        if self.has_label:
            roc_val = roc_auc_score(np.asarray(
                self.val_loader.dataset.label.values, dtype=int), loss_val)
            roc_test = roc_auc_score(np.asarray(
                self.test_loader.dataset.label.values, dtype=int), loss_test)
            print('Epoch: {}, train -log_prob: {:.2f}, test -log_prob: {:.2f}, roc_val: {:.4f}, roc_test: {:.4f}, h: {}'
              .format(self.epoch, np.mean(loss_train), np.mean(loss_val), roc_val, roc_test, self.h.item()))
            self.all_roc_val.append(roc_val)
            self.all_roc_test.append(roc_test)
        else:
            print('Epoch: {}, train -log_prob: {:.2f}, test -log_prob: {:.2f}, h: {}'
              .format(self.epoch, np.mean(loss_train), np.mean(loss_val), self.h.item()))

        # Log the metrics to wandb
        if self.use_wandb and self.has_label:
            wandb.log({
                "Train Loss": np.mean(loss_train),
                "Validation Loss": np.mean(loss_val),
                "Test Loss": np.mean(loss_test),
                "Validation ROC": roc_val,
                "Test ROC": roc_test,
            }, step=self.epoch)
        elif self.use_wandb and not self.has_label:
            wandb.log({
                "Train Loss": np.mean(loss_train),
                "Validation Loss": np.mean(loss_val),
                "Test Loss": np.mean(loss_test),
            }, step=self.epoch)

        if final_epochs:
            if self.has_label:
                self.save_best_checkpoint_by_roc(roc_val)
            else:
                self.save_best_checkpoint_by_loss(loss_val)

    def test(self):
        self.model.eval()
        loss_test = []
        with torch.no_grad():
            for x in self.test_loader:
                x = x.to(self.args.device)
                loss = -self.model.test(x, self.A.data).cpu().numpy()
                loss_test.append(loss)

        loss_test = np.concatenate(loss_test)
        loss_test = np.nan_to_num(loss_test)
        if self.has_label:
            roc_test = roc_auc_score(np.asarray(
                self.test_loader.dataset.label.values, dtype=int), loss_test)
            print(
                'Test -log_prob: {:.2f}, roc_test: {:.4f}'.format(np.mean(loss_test), roc_test))
        else:
            print(
                'Test -log_prob: {:.2f}'.format(np.mean(loss_test)))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.A.data = checkpoint['A_data']
        self.epoch = checkpoint['epoch']
        if 'best_loss' in checkpoint.keys():
            self.best_loss = checkpoint['best_loss']
        else:
            self.roc_val_best = checkpoint['best_roc_score']
        print(f"Loaded model from {checkpoint_path}")

    def load_best_checkpoint(self):
        checkpoint_path = os.path.join(
            self.args.save_path, "{}_best.pt".format(self.args.name))
        self.load_checkpoint(checkpoint_path)

    def save_best_checkpoint_by_loss(self, loss_val):
        if np.mean(loss_val) < self.best_loss and self.args.save_results:
            self.best_loss = np.mean(loss_val)
            print("Save best model so far at {} epoch".format(self.epoch))
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'A_data': self.A.data,
                'epoch': self.epoch,
                'best_loss': self.best_loss
            }
            torch.save(checkpoint, os.path.join(
                self.args.save_path, "{}_best.pt".format(self.args.name)))

    def save_best_checkpoint_by_roc(self, roc_val):
        if np.mean(roc_val) > self.roc_val_best and self.args.save_results:
            self.roc_val_best = np.mean(roc_val)
            print("Save best model so far at {} epoch".format(self.epoch))
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'A_data': self.A.data,
                'epoch': self.epoch,
                'best_roc_score': self.roc_val_best
            }
            torch.save(checkpoint, os.path.join(
                self.args.save_path, "{}_best.pt".format(self.args.name)))
