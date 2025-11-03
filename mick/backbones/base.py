import torch
import logging
from torch import nn
from .__init__ import methods_map

__all__ = ['ModelManager']

class MIA(nn.Module):

    def __init__(self, args):

        super(MIA, self).__init__()

        self.args = args
        fusion_method = methods_map[args.method]
        self.model = fusion_method(args)

    def forward(self, text_feats, video_feats, audio_feats, xReact_comet_feats=None, xWant_comet_feats=None, \
                xReact_sbert_feats=None, xWant_sbert_feats=None, caption_feats=None, xBefore_feats = None, xAfter_feats = None):

        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        
        if self.args.method == 'mick':
            mm_model = self.model(text_feats, video_feats, audio_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats, caption_feats, xBefore_feats, xAfter_feats)
        else:
            mm_model = self.model(text_feats, video_feats, audio_feats)
        return mm_model
        
class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args) 
        model.to(self.device)
        return model