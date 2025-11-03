class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
            'eval_monitor': 'f1',
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            beta_shift (float): The coefficient for nonverbal displacement to create the multimodal vector.
            dropout_prob (float): The embedding dropout probability.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            aligned_method (str): The method for aligning different modalities. ('ctc', 'conv1d', 'avg_pool')
            weight_decay (float): The coefficient for L2 regularization. 
            weight_fuse_relation (float): The weight for fusing relation embeddings.
            weight_fuse_visual_comet (float): The weight for fusing visual comet embeddings.
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'beta_shift': 0.005,
            'dropout_prob': 0.5,
            'warmup_proportion': 0.1,
            'lr': 0.00002,
            'aligned_method': 'ctc',
            'weight_decay': 0.03,
            'temp': 0.07,
            'dst_feature_dims': 768,
            'n_levels_self': 1,
            'n_levels_cross': 1,
            'dropout_rate': 0.2,
            'cross_dp_rate': 0.3,
            'cross_num_heads': 12,
            'self_num_heads': 8,
            # 'grad_clip': 7, 
            # 'opt_patience': 8,
            # 'factor': 0.5,
            # 'aug_lr': 1e-6,
            # 'aug_epoch': 1,
            # 'aug_dp': 0.3,
            # 'aug_weight_decay': 0.1,
            # 'aug_grad_clip': -1.0,
            # 'aug_batch_size': 16,
            # 'aug': True,
            # 'beta_shift': 0.005,
            # 'dropout_prob': 0.5,
            # 'warmup_proportion': 0.1,
        }
        return hyper_parameters 