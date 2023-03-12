# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn
import random
import numpy as np
from modules.encoder import EncoderCNN, EncoderLabels
from modules.transformer_decoder import DecoderTransformer
from modules.quantity_decoder import Quantity_DecoderTransformer, Quantity_MLP
from modules.multihead_attention import MultiheadAttention
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
import pickle
import os
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label2onehot(labels, pad_value):

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    # remove pad position
    one_hot = one_hot[:, :-1]
    # eos position is always 0
    one_hot[:, 0] = 0

    return one_hot


def mask_from_eos(ids, eos_value, mult_before=True):
    mask = torch.ones(ids.size()).to(device).byte()
    mask_aux = torch.ones(ids.size(0)).to(device).byte()

    # find eos in ingredient prediction
    for idx in range(ids.size(1)):
        # force mask to have 1s in the first position to avoid division by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask


def get_model(args, ingr_vocab_size):

    # build ingredients embedding ## EncoderLabels??? ## recipe input으로 들어갈 ingredient embedding이었음
    # encoder_ingrs = EncoderLabels(args.embed_size, ingr_vocab_size,
    #                               args.dropout_encoder, scale_grad=False).to(device)
    # build image model
    encoder_image = EncoderCNN(args.embed_size, args.dropout_encoder, args.image_model)

    # decoder = DecoderTransformer(args.embed_size, instrs_vocab_size,
    #                              dropout=args.dropout_decoder_r, seq_length=args.maxseqlen,
    #                              num_instrs=args.maxnuminstrs,
    #                              attention_nheads=args.n_att, num_layers=args.transf_layers, ## 16
    #                              normalize_before=True,
    #                              normalize_inputs=False,
    #                              last_ln=False,
    #                              scale_embed_grad=False)

    ingr_decoder = DecoderTransformer(args.embed_size, ingr_vocab_size, dropout=args.dropout_decoder_i, ##0.3
                                      seq_length=args.maxnumlabels,
                                      num_instrs=1, attention_nheads=args.n_att_ingrs, ## 4
                                      pos_embeddings=False,
                                      num_layers=args.transf_layers_ingrs, ## 4
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)

    ## quantity_decoder
    # quantity_decoder = Quantity_DecoderTransformer(args.embed_size, output_size = ingr_vocab_size-1, num_layers = 4, nhead = 4, dropout=0.3)
    quantity_decoder = Quantity_MLP(input_size = 512*49, hidden_size = args.quantity_hidden_size, output_size = ingr_vocab_size-1) ## 512 -> 1024 -> 1487 ## TODO hardcode here!!

    # recipe loss
    # criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size-1], reduce=False)

    # ingredients loss
    label_loss = nn.BCELoss(reduce = None)
    eos_loss = nn.BCELoss(reduce = None)
    quantity_loss = nn.MSELoss(reduce = None)

    model = InverseCookingModel(ingr_decoder, quantity_decoder, encoder_image,
                                crit_ingr=label_loss, crit_eos=eos_loss, crit_quantity=quantity_loss,
                                pad_value=ingr_vocab_size-1,
                                ingrs_only=args.ingrs_only, recipe_only=args.recipe_only,
                                label_smoothing=args.label_smoothing_ingr)

    return model


class InverseCookingModel(nn.Module):
    def __init__(self, ingr_decoder, quantity_decoder, image_encoder,
                 crit_ingr=None, crit_eos=None, crit_quantity=None,
                 pad_value=0, ingrs_only=True,
                 recipe_only=False, label_smoothing=0.0):

        super(InverseCookingModel, self).__init__()

        # self.ingredient_encoder = ingredient_encoder
        # self.recipe_decoder = recipe_decoder
        self.image_encoder = image_encoder
        self.ingredient_decoder = ingr_decoder
        # self.crit = crit
        self.crit_ingr = crit_ingr
        self.pad_value = pad_value
        self.ingrs_only = ingrs_only
        self.recipe_only = recipe_only
        self.crit_eos = crit_eos
        self.label_smoothing = label_smoothing

        self.quantity_decoder = quantity_decoder
        self.crit_quantity = crit_quantity

    def forward(self, img_inputs, target_ingrs, target_quantity,
                sample=False, keep_cnn_gradients=False):

        losses = {}
        if sample:
            return self.sample(img_inputs, greedy=True)

        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)
        pred_quantity = self.quantity_decoder(img_features) ## img_feature shape = [38(batch), 512, 49]
        nonzero_mask = (target_quantity > 0).float()
        example_weights = nonzero_mask.sum(dim=1)
        example_weights = example_weights / (example_weights.mean()+1e-8)

        quantity_loss = self.crit_quantity(pred_quantity, target_quantity)
        quantity_loss = (quantity_loss * nonzero_mask).sum(dim=1) / (nonzero_mask.sum(dim=1) + 1e-8)

        quantity_loss = (quantity_loss * example_weights) ## 어차피 quantity info 있는 데이터들만 이용할 것!
        # quantity_loss = quantity_loss.mean()
        # ## target_quantity 가 0이면 (quantity 정도 없으면), quantity_loss = 0
        # no_quantity = torch.sum(target_quantity, dim=1) == 0
        # quantity_loss[no_quantity] = 0
        losses['quantity_loss'] = quantity_loss


        # targets = captions[:, 1:]
        # targets = targets.contiguous().view(-1)

        # img_features = self.image_encoder(img_inputs, keep_cnn_gradients)

        # losses = {}
        target_one_hot = label2onehot(target_ingrs, self.pad_value)
        target_one_hot_smooth = label2onehot(target_ingrs, self.pad_value)

        # ingredient prediction
        target_one_hot_smooth[target_one_hot_smooth == 1] = (1-self.label_smoothing)
        target_one_hot_smooth[target_one_hot_smooth == 0] = self.label_smoothing / target_one_hot_smooth.size(-1)

        # decode ingredients with transformer
        # autoregressive mode for ingredient decoder
        ingr_ids, ingr_logits = self.ingredient_decoder.sample(None, None, greedy=True,
                                                                temperature=1.0, img_features=img_features,
                                                                first_token_value=0, replacement=False)

        ingr_logits = torch.nn.functional.softmax(ingr_logits, dim=-1)

        # find idxs for eos ingredient
        # eos probability is the one assigned to the first position of the softmax
        eos = ingr_logits[:, :, 0]
        target_eos = ((target_ingrs == 0) ^ (target_ingrs == self.pad_value))

        eos_pos = (target_ingrs == 0)
        eos_head = ((target_ingrs != self.pad_value) & (target_ingrs != 0))

        # select transformer steps to pool from
        mask_perminv = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)
        ingr_probs = ingr_logits * mask_perminv.float().unsqueeze(-1)

        ingr_probs, _ = torch.max(ingr_probs, dim=1)

        # ignore predicted ingredients after eos in ground truth
        ingr_ids[mask_perminv == 0] = self.pad_value

        ingr_loss = self.crit_ingr(ingr_probs, target_one_hot_smooth)
        ingr_loss = torch.mean(ingr_loss, dim=-1)

        losses['ingr_loss'] = ingr_loss

        # cardinality penalty
        losses['card_penalty'] = torch.abs((ingr_probs*target_one_hot).sum(1) - target_one_hot.sum(1)) + \
                                    torch.abs((ingr_probs*(1-target_one_hot)).sum(1))

        eos_loss = self.crit_eos(eos, target_eos.float())

        mult = 1/2
        # eos loss is only computed for timesteps <= t_eos and equally penalizes 0s and 1s
        losses['eos_loss'] = mult*(eos_loss * eos_pos.float()).sum(1) / (eos_pos.float().sum(1) + 1e-6) + \
                                mult*(eos_loss * eos_head.float()).sum(1) / (eos_head.float().sum(1) + 1e-6)
        # iou
        pred_one_hot = label2onehot(ingr_ids, self.pad_value)
        # iou sample during training is computed using the true eos position
        losses['iou'] = softIoU(pred_one_hot, target_one_hot)

        return losses

    def sample(self, img_inputs, greedy=True, temperature=1.0, beam=-1, true_ingrs=None):

        outputs = dict()

        img_features = self.image_encoder(img_inputs)

        ingr_ids, ingr_probs = self.ingredient_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                                beam=-1,
                                                                img_features=img_features, first_token_value=0,
                                                                replacement=False)
        
        pred_quantity = self.quantity_decoder(img_features)

        # pred_quantity = np.zeros_like(quantity_output)
        # i = 0 
        # while ingr_ids[i] != 0 and ingr_ids[i] != 1478: ## TODO hardcode 바꾸기
        #     pred_quantity[int(ingr_ids[i])] = quantity_output[int(ingr_ids[i])]
        #     i += 1
        
        outputs['pred_quantity'] = pred_quantity

        # mask ingredients after finding eos
        sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
        ingr_ids[sample_mask == 0] = self.pad_value

        outputs['ingr_ids'] = ingr_ids
        outputs['ingr_probs'] = ingr_probs.data

        mask = sample_mask
        input_mask = mask.float().unsqueeze(1)
        input_feats = self.ingredient_encoder(ingr_ids)


        return outputs


