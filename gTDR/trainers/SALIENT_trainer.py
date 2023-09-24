from pathlib import Path
from typing import List, Dict, Any
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch
import wandb

from gTDR.utils.SALIENT.fast_trainer.utils import (
    Timer,
    setup_runtime_stats,
    report_runtime_stats,
    enable_runtime_stats,
    disable_runtime_stats,
)
from gTDR.utils.SALIENT.drivers import *
from gTDR.utils.SALIENT.drivers.ddp import DDPConfig, set_master, get_ddp_config


class Trainer:
    def __init__(self, model_type, args):
        self.args = args
        self.model_type = model_type
        self.trial_results = []
        self.job_dir = self.get_job_dir()

    def get_job_dir(self):
        return Path(self.args.output_root).joinpath(self.args.job_name)

    def ddp_main(self, rank, dataset, model_type, ddp_cfg: DDPConfig):
        device = torch.device(type='cuda', index=rank)
        drv = DDPDriver(self.args, device, dataset, model_type, ddp_cfg)
        self.drv = drv
        self._train()

    def consume_prefix_in_state_dict_if_present(
        state_dict: Dict[str, Any], prefix: str
    ) -> None:
        r"""Strip the prefix in state_dict in place, if any.

        ..note::
            Given a `state_dict` from a DP/DDP model, a local model can
            load it by applying
            `consume_prefix_in_state_dict_if_present(state_dict,
            "module.")` before calling
            :meth:`torch.nn.Module.load_state_dict`.

        Args:
            state_dict (OrderedDict): a state-dict to be loaded to the model.
            prefix (str): prefix.

        """

        #state_dict = _state_dict.copy()
        keys = sorted(state_dict.keys())
        for key in keys:
            if key.startswith(prefix):
                newkey = key[len(prefix):]
                state_dict[newkey] = state_dict.pop(key)

        # also strip the prefix in metadata if any.
        if "_metadata" in state_dict:
            metadata = state_dict["_metadata"]
            for key in list(metadata.keys()):
                # for the metadata dict, the key can be:
                # '': for the DDP module, which we want to remove.
                # 'module': for the actual model.
                # 'module.xx.xx': for the rest.

                if len(key) == 0:
                    continue
                newkey = key[len(prefix):]
                metadata[newkey] = metadata.pop(key)
        return state_dict

    def _train(self):
        setup_runtime_stats(self.args)

        self.drv.reset()
        # drv.model.module.reset_parameters()
        delta = min(self.args.test_epoch_frequency, self.args.epochs)
        do_eval = self.args.epochs >= self.args.test_epoch_frequency
        best_acc = 0
        self.all_acc = []
        job_dir = self.job_dir
        best_epoch = None

        if self.drv.is_main_proc:
            print()
            print("+" + "-"*40 + "+")
            print("+" + " "*7 + "TRIAL "+"Performing training" + " "*7 + "+")
            print("+" + "-"*40 + "+")

        for epoch in range(0, self.args.epochs, delta):
            if isinstance(self.drv, DDPDriver):
                dist.barrier()
            enable_runtime_stats()
            self.drv.train(range(epoch, epoch + delta))
            disable_runtime_stats()
            if do_eval:
                if isinstance(self.drv, DDPDriver):
                    dist.barrier()
                acc_type = 'valid'
                acc = self.drv.test((acc_type,))[acc_type]
                self.all_acc.append(acc)
                if self.drv.is_main_proc:
                    self.drv.log((acc_type, 'Accurracy', acc))
                if acc > best_acc:
                    best_acc = acc
                    this_epoch = epoch + delta - 1
                    best_epoch = this_epoch
                    if self.drv.is_main_proc:
                        if self.args.save_results:
                            torch.save(
                                self.drv.model.state_dict(),
                                job_dir.joinpath(f'model_{this_epoch}.pt'))
                            with job_dir.joinpath('metadata.txt').open('a') as f:
                                f.write(','.join(map(str, (this_epoch, acc))))
                                f.write('\n')

                if self.drv.is_main_proc:
                    print("Best validation accuracy so far: " + str(best_acc))
                if isinstance(self.drv, DDPDriver):
                    dist.barrier()
                # Log metrics to wandb
                if self.use_wandb:
                    wandb.log({"epoch": epoch, "validation_accuracy": acc})
            self.drv.flush_logs()

        self.best_epoch = best_epoch
        report_runtime_stats(self.drv.log)

        if self.drv.is_main_proc:
            print("Training complete.")
            print("\nModel with best validation accuracy is at " +
                  str(job_dir.joinpath(f'model_{best_epoch}.pt')))

        self.drv.flush_logs()

    def train(self, dataset, use_wandb=False, parameter_search=False):
        self.use_wandb = use_wandb
        if callable(parameter_search):
            self.use_wandb = True
            parameter_search(self)
        if self.args.do_ddp:
            if self.args.total_num_nodes == 1:
                assert self.args.one_node_ddp
                print("Fall into 1 node ddp")
                set_master('localhost')
                ddp_cfg = DDPConfig(0, self.args.num_devices_per_node, 1)
            else:
                ddp_cfg = get_ddp_config(self.args.ddp_dir, self.args.total_num_nodes,
                                         self.args.num_devices_per_node)

            print(f'Using DDP with {ddp_cfg.total_num_nodes} nodes')
            with Timer('dataset.share_memory_()'):
                dataset.share_memory_()

            mp.spawn(self.ddp_main, args=(self.args, dataset, self.model_type, ddp_cfg),
                     nprocs=self.args.num_devices_per_node, join=True)

        else:
            devices = [torch.device(type='cuda', index=i)
                       for i in range(self.args.num_devices_per_node)]
            print(f'Using {self.args.train_type} training')
            drv = SingleProcDriver(
                self.args, devices, dataset, self.model_type)
            self.drv = drv
            self._train()
        if callable(parameter_search):
            self.run.finish()

    def test(self, dataset):
        devices = [torch.device(type='cuda', index=i)
                   for i in range(self.args.num_devices_per_node)]
        drv = SingleProcDriver(self.args, devices, dataset, self.model_type)
        acc_type = 'test'
        acc_test = self.drv.test((acc_type,))[acc_type]
        if isinstance(drv, DDPDriver):
            dist.barrier()
        if drv.is_main_proc:
            print("Test accuracy is: " + str(acc_test))

    def load_best_checkpoint(self):
        self.drv.model.load_state_dict(torch.load(
            self.job_dir.joinpath(f'model_{self.best_epoch}.pt')))
        print(
            f"Loaded model from {self.job_dir.joinpath(f'model_{self.best_epoch}.pt')}")

    def load_checkpoint(self, path: str):
        """Load model state from a checkpoint file"""
        if Path(path).exists():
            checkpoint = torch.load(path)
            self.drv.model.load_state_dict(
                self.consume_prefix_in_state_dict_if_present(checkpoint, 'module.'))
            print(f"Loaded model from {path}")
        else:
            print(f"No checkpoint found at {path}")

    def visualize_validation(self):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot(self.all_acc, color="#0072B2",
                alpha=1.0, lw=4.0, label='val acc')
        ax.set_xlabel('Iteration', labelpad=5)
        ax.set_ylabel(f'Log(acc)', labelpad=5)
        plt.yscale('log')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

        if self.args.save_results:
            plt.savefig(f"{self.args.output_root}/{self.args.dataset_name}_val_hist.png",
                        transparent=True, bbox_inches='tight', pad_inches=0)
