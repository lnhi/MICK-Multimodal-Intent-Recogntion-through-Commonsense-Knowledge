from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPooler
import torch.nn.functional as F
from torch import nn
import torch
from ..SubNets.AlignNets import AlignSubNet
from ..SubNets.FeatureNets import BERTEncoder, BertCrossEncoder

class CommonsenseTextEncoder(nn.Module):
    def __init__(self, args):
        super(CommonsenseTextEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(args.text_backbone)

    def get_embedding(self, text):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state  
        sequence_embedding = torch.mean(word_embeddings, dim=1) 

        return word_embeddings
    
    def forward(self, text_feats, xReact_comet, xWant_comet, xReact_sbert, xWant_sbert, caption_feats, xAfter_feats, xBefore_feats):
        text_emb = self.get_embedding(text_feats)
    
        xReact_comet_emb = self.get_embedding(xReact_comet)
        xWant_comet_emb = self.get_embedding(xWant_comet)
        xReact_sbert_emb = self.get_embedding(xReact_sbert)
        xWant_sbert_emb = self.get_embedding(xWant_sbert)
        caption_emb = self.get_embedding(caption_feats)
        xBefore_emb = self.get_embedding(xBefore_feats)
        xAfter_emb = self.get_embedding(xAfter_feats)

        return text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb, caption_emb, xBefore_emb, xAfter_emb
    
class CommonsenseInferenceFusion(nn.Module):
    def __init__(self, args):
        super(CommonsenseInferenceFusion, self).__init__()
        self.linear_layer = nn.Linear(args.text_feat_dim + args.relation_feat_dim * 2, args.text_feat_dim)
    
    def forward(self, text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb, xBefore_emb, xAfter_emb):
        xReact_encoder_outputs_utt = torch.cat((text_emb, xReact_comet_emb, xReact_sbert_emb), -1)
        xWant_encoder_outputs_utt = torch.cat((text_emb, xWant_comet_emb, xWant_sbert_emb), -1)
        xVisual_comet_encoder_utt = torch.cat((text_emb, xBefore_emb, xAfter_emb), -1)
        indicator_r = self.linear_layer(xReact_encoder_outputs_utt)
        indicator_w = self.linear_layer(xWant_encoder_outputs_utt)
        indicator_v = self.linear_layer(xVisual_comet_encoder_utt)
        
        indicator_r_ = F.softmax(indicator_r, dim=-1)
        indicator_w_ = F.softmax(indicator_w, dim=-1)
        indicator_v_ = F.softmax(indicator_v, dim=-1)

        indicator_r_ = indicator_r_[:, :, 0].unsqueeze(2).repeat(1, 1, text_emb.size(-1))
        indicator_w_ = indicator_w_[:, :, 0].unsqueeze(2).repeat(1, 1, text_emb.size(-1))
        indicator_v_ = indicator_v_[:, :, 0].unsqueeze(2).repeat(1, 1, text_emb.size(-1))

        new_xReact_encoder_outputs_utt = indicator_r_ * xReact_comet_emb + (1 - indicator_r_) * xReact_sbert_emb
        new_xWant_encoder_outputs_utt = indicator_w_ * xWant_comet_emb + (1 - indicator_w_) * xWant_sbert_emb
        new_xVisual_comet_encoder_utt = indicator_v_ * xBefore_emb + (1 - indicator_v_) * xAfter_emb

        return new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt, new_xVisual_comet_encoder_utt
    
class TextualCommonsenseEnrichment(nn.Module):
    def __init__(self, args):
        super(TextualCommonsenseEnrichment, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(args.text_feat_dim))
        self.W.requires_grad = True
        self.alpha = args.weight_fuse_relation

    def forward(self, text_emb, xReact_emb, xWant_emb):
        z1 = text_emb + self.W * xReact_emb
        z2 = text_emb + self.W * xWant_emb

        z = self.alpha * z1 + (1 - self.alpha) * z2

        return z

class VisualCommonsenseEnrichment(nn.Module):
    def __init__(self, args):
        super(VisualCommonsenseEnrichment, self).__init__()
        self.beta = args.weight_fuse_visual_comet
    
    def forward(self, zT, new_xVisual_comet_encoder_utt):
        z = self.beta * zT + (1 - self.beta) * new_xVisual_comet_encoder_utt
        return z

class MFA(nn.Module):
    
    def __init__(self, args):

        super(MFA, self).__init__()
        self.args = args

        self.visual_size = args.video_feat_dim
        self.acoustic_size = args.audio_feat_dim
        self.text_size = args.text_feat_dim
        self.device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dst_feature_dims = args.dst_feature_dims

        self.layers_cross = args.n_levels_cross 
        self.layers_self = args.n_levels_self 

        self.dropout_rate = args.dropout_rate
        self.cross_dp_rate = args.cross_dp_rate
        self.cross_num_heads = args.cross_num_heads
        self.self_num_heads = args.self_num_heads

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dst_feature_dims, nhead=self.self_num_heads)
        self.self_att = nn.TransformerEncoder(encoder_layer, num_layers=self.layers_self)

        self.video2text_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        self.audio2text_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        
        self.v2t_project = nn.Linear(self.visual_size, self.text_size)

        self.mlp_project =  nn.Sequential(
                nn.Linear(self.dst_feature_dims*3, self.dst_feature_dims),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

    def forward(self, text_feats, video_feats, audio_feats, text_mask):
        # first layer : T,V,A
        bert_sent_mask = text_mask
        text_seq = text_feats  

        video_seq = self.v2t_project(video_feats)
        audio_seq = audio_feats

        video_mask = torch.sum(video_feats.ne(torch.zeros(video_feats[0].shape[-1]).to(self.device)).int(), dim=-1)/video_feats[0].shape[-1]
        video_mask_len = torch.sum(video_mask, dim=1, keepdim=True)  

        video_mask_len = torch.where(video_mask_len > 0.5, video_mask_len, torch.ones([1]).to(self.device))
        video_masked_output = torch.mul(video_mask.unsqueeze(2), video_seq)
        video_rep = torch.sum(video_masked_output, dim=1, keepdim=False) / video_mask_len
        
        audio_mask = torch.sum(audio_feats.ne(torch.zeros(audio_feats[0].shape[-1]).to(self.device)).int(), dim=-1)/audio_feats[0].shape[-1]
        audio_mask_len = torch.sum(audio_mask, dim=1, keepdim=True)  
        
        audio_masked_output = torch.mul(audio_mask.unsqueeze(2), audio_seq)
        audio_rep = torch.sum(audio_masked_output, dim=1, keepdim=False) / audio_mask_len
        
        # second layer (V,A) --> T: V_T, A_T
        extended_video_mask = video_mask.unsqueeze(1).unsqueeze(2)
        extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype)
        extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        video2text_seq = self.video2text_cross(text_seq, video_seq, extended_video_mask)

        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0
        audio2text_seq = self.audio2text_cross(text_seq, audio_seq, extended_audio_mask)

        text_mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) 

        video2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), video2text_seq)
        video2text_rep = torch.sum(video2text_masked_output, dim=1, keepdim=False) / text_mask_len
        
        audio2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), audio2text_seq)
        audio2text_rep = torch.sum(audio2text_masked_output, dim=1, keepdim=False) / text_mask_len
        
        # third layer: mlp->VAL
        shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=2))

        return shallow_seq

class MICK(nn.Module):
    def __init__(self, args):
        super(MICK, self).__init__()
        self.config = BertConfig.from_pretrained(args.text_backbone)
        self.text_encoder = CommonsenseTextEncoder(args)
        self.commonsense_fusion = CommonsenseInferenceFusion(args)
        self.textual_fusion = TextualCommonsenseEnrichment(args)
        self.visual_fusion = VisualCommonsenseEnrichment(args)
        self.mfa = MFA(args)

        self.pooler = BertPooler(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)
        
        self.temp = args.temp

    def contrastive_loss(self, feats_1, feats_2):
        feats_1 = feats_1.mean(dim=1)
        feats_2 = feats_2.mean(dim=1 )
        sim_matrix = torch.matmul(feats_1, feats_2.T)

        i_logsoftmax = nn.functional.log_softmax(sim_matrix / self.temp, dim=1)
        j_logsoftmax = nn.functional.log_softmax(sim_matrix.T / self.temp, dim=1)
    
        i_diag = torch.diag(i_logsoftmax)
        loss_i = i_diag.mean()

        j_diag = torch.diag(j_logsoftmax)
        loss_j = j_diag.mean()

        con_loss = - (loss_i + loss_j) / 2 
        return con_loss
    
    def forward(self, text_feats, video_feats, audio_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats, caption_feats, xBefore_feats, xAfter_feats):
        text_mask = text_feats[:, 1]
        
        text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb, caption_emb, xBefore_emb, xAfter_emb = \
            self.text_encoder(text_feats, xReact_comet_feats, xWant_comet_feats, xReact_sbert_feats, xWant_sbert_feats, caption_feats, xBefore_feats, xAfter_feats)

        new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt, new_xVisual_comet_outputs_utt = self.commonsense_fusion(text_emb, xReact_comet_emb, xWant_comet_emb, xReact_sbert_emb, xWant_sbert_emb, xBefore_emb, xAfter_emb)

        text_feats_no_caption = self.textual_fusion(text_emb, new_xReact_encoder_outputs_utt, new_xWant_encoder_outputs_utt)
        text_with_visual_comet = self.visual_fusion(text_feats_no_caption, new_xVisual_comet_outputs_utt)

        output = self.pooler(self.mfa(text_with_visual_comet, video_feats, audio_feats, text_mask))

        output = self.dropout(output)
        logits = self.classifier(output)
  
        return logits