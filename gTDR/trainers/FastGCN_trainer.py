import time
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from gTDR.models import FastGCN
import gTDR.utils.FastGCN as fastgcn_utils
from sklearn.metrics import f1_score as F1
import os


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.X = model.X
        self.y = model.y
        self.data = model.data
        self.adjmat = model.adjmat

        self.training_mask = self.data.train_mask
        self.training_indices = torch.where(
            self.training_mask == True)[0].cpu().numpy()
        self.stochastic = args.fast

        self.optimizer = torch.optim.Adam(params=self.model.parameters(
        ), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.criteria = nn.NLLLoss(reduction='mean')

        self.batches = [self.args.init_batch] + [min(self.X.shape[0], int(self.args.sample_size * (
            1 if i == 0 else self.args.scale_factor))) for i in range(len(self.model.layers) - 1)]
        self.inference_batches = [self.args.inference_init_batch] + [min(self.X.shape[0], int(
            self.args.inference_sample_size * (1 if i == 0 else self.args.scale_factor))) for i in range(len(self.model.layers) - 1)]

        self.loss_hist = []
        self.val_hist = []
        self.test_acc = []

        self.best_val_score = float('inf')  # Set it to +infinity initially
        # The path to save the best model
        #self.best_model_path = f"{self.args.save_path}/{self.args.dataset}_best_model.pt"
        self.best_model_path = os.path.join(self.args.save_path, f"{self.args.dataset}_best_model.pt")
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        self.cs = {
            "orange": "#E69F00",
            "blue": "#0072B2",
                    "red": "#D55E00",
        }

    def train(self, use_wandb=False, parameter_search=False):
        self.use_wandb = use_wandb
        if callable(parameter_search):
            self.use_wandb = True
            parameter_search(self)

        print(f"{'=' * 25} STARTING TRAINING {'=' * 25}")
        print(f"TRAINING INFORMATION:")
        print(f"[DATA] {self.args.dataset} dataset")
        print(f"[FAST] using FastGCN? {self.args.fast}")
        print(
            f"[INF] using sampling for inference? {self.args.samp_inference}")
        print(f"[DEV] device: {self.args.device}")
        print(f"[ITERS] performing {self.args.epochs} Adam updates")
        print(f"[LR] Adam learning rate: {self.args.lr}")

        if self.args.fast:
            print(f"[BATCH] batch size: {self.batches}")

        # max_acc = 0
        running_time = 0
        total_times = []
        for i in range(1, self.args.epochs + 1):
            t0 = time.time()
            self.loss_hist = fastgcn_utils.train(self.model, self.optimizer, self.X, self.y, self.adjmat,
                                                 self.training_mask, self.criteria, self.stochastic, self.loss_hist, self.batches,
                                                 self.training_indices)
            # Log training loss to wandb
            if self.use_wandb:
                wandb.log({"Train Loss": self.loss_hist[-1]}, step=i)
            t1 = time.time()
            running_time += t1 - t0
            if i > 0:
                total_times.append(t1 - t0)

            if i % self.args.report == 0 and i > 1:
                self.test_acc = fastgcn_utils.test(
                    self.model, self.X, self.y, self.adjmat, self.data.test_mask, self.test_acc)
                # Log test accuracy to wandb
                if self.use_wandb:
                    wandb.log({"Test Accuracy": self.test_acc[-1]}, step=i)
                if self.args.use_val:
                    self.val_hist = fastgcn_utils.validation_test(
                        self.model, self.X, self.y, self.adjmat, self.data.val_mask, self.criteria, self.val_hist)
                    # The most recent validation score
                    current_val_score = self.val_hist[-1]
                    # Log validation loss to wandb
                    if self.use_wandb:
                        wandb.log(
                            {"Validation Loss": current_val_score}, step=i)
                    if current_val_score < self.best_val_score:
                        self.best_val_score = current_val_score
                        if self.args.save_results:
                            self.save_best_checkpoint()

                if len(self.val_hist) > self.args.early_stop:
                    if self.val_hist[-(self.args.early_stop + 1)] <= min(self.val_hist[-self.args.early_stop:]):
                        print(f"[STOP] early stopping at iteration: {i}\n")
                        break

        print(f"RESULTS:")
        print(f"[LOSS] minimum loss: {min(self.loss_hist)}")
        print(
            f"[BATCH TIME] {round(sum(total_times) / len(total_times), 4)} seconds")
        print(f"[TOTAL TIME] {round(running_time, 4)} seconds")
        print(f"{'=' * 26} ENDING TRAINING {'=' * 26}\n")
        if callable(parameter_search):
            self.run.finish()

    def test(self):
        # Set to test
        self.model.eval()

        # Forward pass
        out, _ = self.model.predict(self.X, self.adjmat)
        out = out[self.data.test_mask].cpu().numpy()
        y = self.y[self.data.test_mask].cpu().numpy()
        acc = F1(y, out, average='micro') * 100.0
        print(f"[ACC] micro F1 testing accuracy: {acc} %")

    def save_best_checkpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.best_model_path)

    def load_best_checkpoint(self):
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("~~~ Loaded best checkpoint ~~~")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("~~~ Loaded given checkpoint ~~~")

    def visualize_train(self):
        """At training time"""
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot(self.loss_hist,
                color=self.cs["blue"], alpha=1.0, lw=4.0, label='train loss')
        ax.plot(self.val_hist, color=self.cs["red"],
                alpha=1.0, lw=4.0, label='val loss')
        ax.set_xlabel('Iteration', labelpad=5)
        ax.set_ylabel(f'Log(loss)', labelpad=5)
        plt.yscale('log')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

        if self.args.save_results:
            plt.savefig(f"{self.args.save_path}/{self.args.dataset}_train_loss.png",
                        transparent=True, bbox_inches='tight', pad_inches=0)

    def visualize_test(self):
        """At testing time"""
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot(self.test_acc,
                color=self.cs["orange"], alpha=1.0, lw=4.0, label='FastGCN')
        ax.set_xlabel('Iteration', labelpad=5)
        ax.set_ylabel(f'Test accuracy', labelpad=5)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

        if self.args.save_results:
            plt.savefig(f"{self.args.save_path}/{self.args.dataset}_testing_accuracy.png",
                        transparent=True, bbox_inches='tight', pad_inches=0)
