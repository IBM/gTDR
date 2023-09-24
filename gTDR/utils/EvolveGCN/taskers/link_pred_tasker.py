import torch
import gTDR.utils.EvolveGCN.taskers_utils as taskers_utils



class Link_Pred_Tasker():
	'''
	Creates a tasker object which computes the required inputs for training on a link prediction
	task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
	makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
	structure).

	Based on the dataset it implements the get_sample function required by edge_cls_trainer.
	This is a dictionary with:
		- time_step: the time_step of the prediction
		- hist_adj_list: the input adjacency matrices until t, each element of the list 
						 is a sparse tensor with the current edges. For link_pred they're
						 unweighted
		- nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
						  two dimmensions: node_idx and node_feats
		- label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2 
					 matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
					 , 0 if it doesn't

	There's a test difference in the behavior, on test (or development), the number of sampled non existing 
	edges should be higher.
	'''
	def __init__(self,args,dataset):
		self.data = dataset
		#max_time for link pred should be one before
		self.max_time = dataset.max_time - 1
		self.args = args
		self.num_classes = 2

		if not (args.use_2_hot_node_feats or args.use_1_hot_node_feats):
			self.feats_per_node = dataset.feats_per_node

		self.get_node_feats = self.build_get_node_feats(args,dataset)
		self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)
		
	def build_prepare_node_feats(self,args,dataset):
		if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
			def prepare_node_feats(node_feats):
				return taskers_utils.sparse_prepare_tensor(node_feats,
											   torch_size= [dataset.num_nodes,
											   				self.feats_per_node])
		else:
			prepare_node_feats = self.data.prepare_node_feats

		return prepare_node_feats


	def build_get_node_feats(self,args,dataset):
		if args.use_2_hot_node_feats:
			max_deg_out, max_deg_in = taskers_utils.get_max_degs(args,dataset)
			self.feats_per_node = max_deg_out + max_deg_in
			def get_node_feats(adj):
				return taskers_utils.get_2_hot_deg_feats(adj,
											  max_deg_out,
											  max_deg_in,
											  dataset.num_nodes)
		elif args.use_1_hot_node_feats:
			max_deg,_ = taskers_utils.get_max_degs(args,dataset)
			self.feats_per_node = max_deg
			def get_node_feats(adj):
				return taskers_utils.get_1_hot_deg_feats(adj,
											  max_deg,
											  dataset.num_nodes)
		else:
			def get_node_feats(adj):
				return dataset.nodes_feats

		return get_node_feats


	def get_sample(self,idx,test, **kwargs):
		hist_adj_list = []
		hist_ndFeats_list = []
		hist_mask_list = []
		existing_nodes = []
		for i in range(idx - self.args.num_hist_steps, idx+1):
			cur_adj = taskers_utils.get_sp_adj(edges = self.data.edges, 
								   time = i,
								   weighted = True,
								   time_window = self.args.adj_mat_time_window)

			if self.args.smart_neg_sampling:
				existing_nodes.append(cur_adj['idx'].unique())
			else:
				existing_nodes = None

			node_mask = taskers_utils.get_node_mask(cur_adj, self.data.num_nodes)

			node_feats = self.get_node_feats(cur_adj)

			cur_adj = taskers_utils.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes)

			hist_adj_list.append(cur_adj)
			hist_ndFeats_list.append(node_feats)
			hist_mask_list.append(node_mask)

		# This would be if we were training on all the edges in the time_window
		label_adj = taskers_utils.get_sp_adj(edges = self.data.edges, 
								  time = idx+1,
								  weighted = False,
								  time_window =  self.args.adj_mat_time_window)
		if test:
			neg_mult = self.args.negative_mult_test
		else:
			neg_mult = self.args.negative_mult_training
			
		if self.args.smart_neg_sampling:
			existing_nodes = torch.cat(existing_nodes)

		
		if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
			non_exisiting_adj = taskers_utils.get_all_non_existing_edges(adj = label_adj, tot_nodes = self.data.num_nodes)
		else:
			non_exisiting_adj = taskers_utils.get_non_existing_edges(adj = label_adj,
													  number = label_adj['vals'].size(0) * neg_mult,
													  tot_nodes = self.data.num_nodes,
													  smart_sampling = self.args.smart_neg_sampling,
													  existing_nodes = existing_nodes)
		
		label_adj['idx'] = torch.cat([label_adj['idx'],non_exisiting_adj['idx']])
		label_adj['vals'] = torch.cat([label_adj['vals'],non_exisiting_adj['vals']])
		return {'idx': idx,
				'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp': label_adj,
				'node_mask_list': hist_mask_list}
