# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import random

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, sample_topk_ratio=1/3., score_pred_net='2layer-fc-256', kproj_net='2layer-fc', unsample_abstract_number=30, pos_embed_kproj=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.sampler = SortSampler(sample_topk_ratio, d_model, score_pred_net=score_pred_net, kproj_net=kproj_net, unsample_abstract_number=unsample_abstract_number, pos_embed_kproj=pos_embed_kproj)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, sample_ratio):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # sample_weight = self.sample_weight_conv(src).sigmoid().view(-1)
        # sort_confidence_topk = sample_weight.sort(descending=True)[1][:self.topk]
        # src = src.flatten(2).permute(2, 0, 1)
        ## reg sample weight to be sparse with l1 loss
        # sample_reg_loss = sample_weight[sort_confidence_topk].mean()
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        src, sample_reg_loss, sort_confidence_topk, mask, pos_embed = self.sampler(src, mask, pos_embed, sample_ratio)
        # src, mask, pos_embed = src[sort_confidence_topk]*sample_weight[sort_confidence_topk].unsqueeze(-1).unsqueeze(-1), mask[:, sort_confidence_topk], pos_embed[sort_confidence_topk]
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), sample_reg_loss


class SortSampler(nn.Module):

    def __init__(self, topk_ratio, input_dim, score_pred_net='2layer-fc', kproj_net='2layer-fc', unsample_abstract_number=30,pos_embed_kproj=False):
        super().__init__()
        self.topk_ratio = topk_ratio
        if score_pred_net == '2layer-fc-256':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(input_dim, 1, 1))
        elif score_pred_net == '2layer-fc-16':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, 16, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(16, 1, 1))
        elif score_pred_net == '1layer-fc':
            self.score_pred_net = nn.Conv2d(input_dim, 1, 1)
        else:
            raise ValueError

        self.norm_feature = nn.LayerNorm(input_dim,elementwise_affine=False)
        self.unsample_abstract_number = unsample_abstract_number
        if kproj_net == '2layer-fc':
            self.k_proj = nn.Sequential(nn.Linear(input_dim, input_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, unsample_abstract_number))
        elif kproj_net == '1layer-fc':
            self.k_proj = nn.Linear(input_dim, unsample_abstract_number)
        else:
            raise ValueError
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.pos_embed_kproj = pos_embed_kproj

    def forward(self, src, mask, pos_embed, sample_ratio):
        bs,c ,h, w  = src.shape
        sample_weight = self.score_pred_net(src).sigmoid().view(bs,-1)
        # sample_weight[mask] = sample_weight[mask].clone() * 0.
        # sample_weight.data[mask] = 0.
        sample_weight_clone = sample_weight.clone().detach()
        sample_weight_clone[mask] = -1.

        if sample_ratio==None:
            sample_ratio = self.topk_ratio
        ##max sample number:
        sample_lens = ((~mask).sum(1)*sample_ratio).int()
        max_sample_num = sample_lens.max()
        mask_topk = torch.arange(max_sample_num).expand(len(sample_lens), max_sample_num).to(sample_lens.device) > (sample_lens-1).unsqueeze(1)

        ## for sampling remaining unsampled points
        min_sample_num = sample_lens.min()

        sort_order = sample_weight_clone.sort(descending=True,dim=1)[1]
        sort_confidence_topk = sort_order[:,:max_sample_num]
        sort_confidence_topk_remaining = sort_order[:,min_sample_num:]
        ## flatten for gathering
        src = src.flatten(2).permute(2, 0, 1)
        src = self.norm_feature(src)

        src_sample_remaining = src.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))

        ## this will maskout the padding and sampled points
        mask_unsampled = torch.arange(mask.size(1)).expand(len(sample_lens), mask.size(1)).to(sample_lens.device) < (sample_lens).unsqueeze(1)
        mask_unsampled = mask_unsampled | mask.gather(1, sort_order)
        mask_unsampled = mask_unsampled[:,min_sample_num:]

        ## abstract the unsampled points with attention
        if self.pos_embed_kproj:
            pos_embed_sample_remaining = pos_embed.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))
            kproj = self.k_proj(src_sample_remaining+pos_embed_sample_remaining)
        else:
            kproj = self.k_proj(src_sample_remaining)
        kproj = kproj.masked_fill(
            mask_unsampled.permute(1,0).unsqueeze(2),
            float('-inf'),
        ).permute(1,2,0).softmax(-1)
        abs_unsampled_points = torch.bmm(kproj, self.v_proj(src_sample_remaining).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_pos_embed = torch.bmm(kproj, pos_embed.gather(0,sort_confidence_topk_remaining.
                                                                          permute(1,0)[...,None].expand(-1,-1,c)).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_mask = mask.new_zeros(mask.size(0),abs_unsampled_points.size(0))

        ## reg sample weight to be sparse with l1 loss
        sample_reg_loss = sample_weight.gather(1,sort_confidence_topk).mean()
        src_sampled = src.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c)) *sample_weight.gather(1,sort_confidence_topk).permute(1,0).unsqueeze(-1)
        pos_embed_sampled = pos_embed.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c))
        mask_sampled = mask_topk

        src = torch.cat([src_sampled, abs_unsampled_points])
        pos_embed = torch.cat([pos_embed_sampled,abs_unsampled_pos_embed])
        mask = torch.cat([mask_sampled, abs_unsampled_mask],dim=1)
        assert ((~mask).sum(1)==sample_lens+self.unsample_abstract_number).all()
        return src, sample_reg_loss, sort_confidence_topk, mask, pos_embed

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        sample_topk_ratio=args.sample_topk_ratio,
        score_pred_net=args.score_pred_net,
        kproj_net=args.kproj_net,
        unsample_abstract_number=args.unsample_abstract_number,
        pos_embed_kproj=args.pos_embed_kproj
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
