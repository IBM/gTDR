import torch
import gTDR.utils.EvolveGCN as egcn_utils
from gTDR.utils.EvolveGCN import logger
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gTDR.utils.EvolveGCN.Cross_Entropy as ce
from torch.utils.data import Dataset, DataLoader
import wandb

class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''
    def __init__(self,args,tasker):
        #### For datsets with time
        assert args.train_proportion + args.dev_proportion < 1, \
            'there\'s no space for test samples'
        #only the training one requires special handling on start, the others are fine with the split IDX.
        start = tasker.data.min_time + args.num_hist_steps #-1 + args.adj_mat_time_window
        end = args.train_proportion
        
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        train = data_split(tasker, start, end, test = False)
        train = DataLoader(train,**args.data_loading_params)

        start = end
        end = args.dev_proportion + args.train_proportion
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        if args.task == 'link_pred':
            dev = data_split(tasker, start, end, test = True, all_edges=True)
        else:
            dev = data_split(tasker, start, end, test = True)

        dev = DataLoader(dev,num_workers=args.data_loading_params['num_workers'])
        
        start = end
        
        #the +1 is because I assume that max_time exists in the dataset
        end = int(tasker.max_time) + 1
        if args.task == 'link_pred':
            test = data_split(tasker, start, end, test = True, all_edges=True)
        else:
            test = data_split(tasker, start, end, test = True)
            
        test = DataLoader(test,num_workers=args.data_loading_params['num_workers'])

        print ('Dataset splits sizes:  train',len(train), 'dev',len(dev), 'test',len(test))
        
        
        self.tasker = tasker
        self.train = train
        self.dev = dev
        self.test = test
        


class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end-self.start

    def __getitem__(self,idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test = self.test, **self.kwargs)
        return t


class Trainer():
    def __init__(self, args, model):
        self.args = args
        dataset = model.dataset
        self.splitter = splitter(args, model.tasker)
        self.tasker = model.tasker
        self.gcn = model
        self.classifier = egcn_utils.build_classifier(args, model.tasker)
        self.comp_loss = ce.Cross_Entropy(args, dataset).to(args.device)

        self.num_nodes = dataset.num_nodes
        self.data = model.dataset
        self.num_classes = model.num_classes

        self.logger = logger.Logger(args, self.num_classes)

        self.init_optimizers(args)

        self.train_metrics_hist = []
        self.val_metrics_hist = []
        self.cs = {
                    "orange": "#E69F00",
                    "blue": "#0072B2",
                    "red": "#D55E00",
                }
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

    def init_optimizers(self,args):
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

    def save_best_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'gcn_dict': self.gcn.state_dict(),
            'classifier_dict': self.classifier.state_dict(),
            'gcn_optimizer': self.gcn_opt.state_dict(),
            'classifier_optimizer': self.classifier_opt.state_dict()
        }
        torch.save(checkpoint, f"{self.args.save_path}/checkpoint.pth.tar")
        print("=> saved best checkpoint so far at '{}' (epoch {})".format(f"{self.args.save_path}/checkpoint.pth.tar", epoch))

    def load_best_checkpoint(self):
        filename = f"{self.args.save_path}/checkpoint.pth.tar"
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            self.gcn.load_state_dict(checkpoint['gcn_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_dict'])
            self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
            self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
            print("=> loaded best checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return epoch
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            return 0
        
    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            self.gcn.load_state_dict(checkpoint['gcn_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_dict'])
            self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
            self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return epoch
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            return 0

    def train(self, parameter_search=False, use_wandb=False):
        self.use_wandb = use_wandb
        
        if callable(parameter_search):
            self.use_wandb = True
            parameter_search(self)
            
        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0
        epochs_without_impr = 0

        for e in range(self.args.num_epochs):
            eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
            self.train_metrics_hist.append(eval_train)
            if len(self.splitter.dev)>0:
                eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
                self.val_metrics_hist.append(eval_valid)
                if self.use_wandb: 
                    wandb.log({f"train_{self.args.target_measure}": eval_train, f"val_{self.args.target_measure}": eval_valid})  # Log metrics
                if eval_valid>best_eval_valid:
                    best_eval_valid = eval_valid
                    if self.args.save_results:
                        self.save_best_checkpoint(e)
                    epochs_without_impr = 0
                    print ('### ep '+str(e)+' - Best valid measure:'+str(eval_valid))
                else:
                    epochs_without_impr+=1
                    if epochs_without_impr>self.args.early_stop_patience:
                        print ('### ep '+str(e)+' - Early stop.')
                        if callable(parameter_search):
                            run.finish()
                        break
        if callable(parameter_search):
            run.finish()

    def test(self):
        log_interval=1
        self.logger.log_epoch_start('TEST', len(self.splitter.test), 'TEST', minibatch_log_interval=log_interval)
        torch.set_grad_enabled(False)
        for s in self.splitter.test:
            s = self.prepare_sample(s)

            predictions, nodes_embs = self.predict(s.hist_adj_list,
                                                   s.hist_ndFeats_list,
                                                   s.label_sp['idx'],
                                                   s.node_mask_list)
            
            loss = self.comp_loss(predictions,s.label_sp['vals'])
            if self.args.task == 'link_pred':
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
            else:
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())

        torch.set_grad_enabled(True)
        eval_measure = self.logger.log_epoch_done()

        return eval_measure, nodes_embs
    
    def visualize_train(self):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot(self.train_metrics_hist, color=self.cs["blue"], alpha=1.0, lw=4.0, label='train')
        ax.plot(self.val_metrics_hist, color=self.cs["red"], alpha=1.0, lw=4.0, label='val')
        ax.set_xlabel('Iteration', labelpad=5)
        ax.set_ylabel(f'Log({self.args.target_measure})', labelpad=5)
        plt.yscale('log')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        # Set the x-axis major locator to integer values
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

        if self.args.save_results:
            plt.savefig(f"{self.args.save_path}/train_val_{self.args.target_measure}.png", transparent=True, bbox_inches='tight', pad_inches=0)
        
    def run_epoch(self, split, epoch, set_name, grad):
        log_interval=999
        if set_name=='TEST':
            log_interval=1
        self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

        torch.set_grad_enabled(grad)
        for s in split:
            s = self.prepare_sample(s)

            predictions, nodes_embs = self.predict(s.hist_adj_list,
                                                   s.hist_ndFeats_list,
                                                   s.label_sp['idx'],
                                                   s.node_mask_list)
            
            loss = self.comp_loss(predictions,s.label_sp['vals'])
            if set_name == 'VALID' and self.use_wandb:
                wandb.log({"val_loss": loss.item()})
            if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
            else:
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
            if grad:
                self.optim_step(loss)

        torch.set_grad_enabled(True)
        eval_measure = self.logger.log_epoch_done()

        return eval_measure, nodes_embs

    def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
        nodes_embs = self.gcn(hist_adj_list,
                              hist_ndFeats_list,
                              mask_list)

        predict_batch_size = 100000
        gather_predictions=[]
        for i in range(1 +(node_indices.size(1)//predict_batch_size)):
            cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
            predictions = self.classifier(cls_input)
            gather_predictions.append(predictions)
        gather_predictions=torch.cat(gather_predictions, dim=0)
        return gather_predictions, nodes_embs

    def gather_node_embs(self,nodes_embs,node_indices):
        cls_input = []

        for node_set in node_indices:
            cls_input.append(nodes_embs[node_set])
        return torch.cat(cls_input,dim = 1)

    def optim_step(self,loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()

        if self.use_wandb: 
            wandb.log({"train_loss": loss.item()})


    def prepare_sample(self,sample):
        sample = egcn_utils.Namespace(sample)
        for i,adj in enumerate(sample.hist_adj_list):
            adj = egcn_utils.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
            node_mask = sample.node_mask_list[i]
            sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer
        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
        else:
            label_sp['idx'] = label_sp['idx'].to(self.args.device)

        label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self,adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj['idx'] = adj['idx'][0]
        adj['vals'] = adj['vals'][0]
        return adj

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

            csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

        pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')