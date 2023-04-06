# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn
import random
import numpy as np
from modules.encoder import EncoderCNN, EncoderLabels, EncoderViT
from modules.transformer_decoder import DecoderTransformer
from modules.multihead_attention import MultiheadAttention
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
import pickle
import os
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

class Inverse_Quantity(nn.Module):
    def __init__(self, inverse_model, quantity_model, quantity_only, train_ingr_only, weighted_loss):
        super().__init__()

        self.inverse_model = inverse_model
        self.quantity_model = quantity_model
        self.quantity_criterion = nn.MSELoss(reduce = False)
        self.quantity_criterion = self.quantity_criterion.to(device)
        self.quantity_only = quantity_only
        self.train_ingr_only = train_ingr_only
        self.weighted_loss = weighted_loss

    def forward(self, img_inputs, ingr_gt, quantity_gt, class_gt, sample = False, keep_cnn_gradients=False):
        
        if sample:
            outputs = self.inverse_model.sample(img_inputs, greedy=True)
            if self.train_ingr_only:
                return outputs

            img_embedding = self.inverse_model.image_encoder(img_inputs.to(device), keep_cnn_gradients = False)
            pred_quantity = self.quantity_model(img_embedding)
            outputs['quantity'] = pred_quantity
            return outputs    
        
        img_inputs = img_inputs.to(device)
        ingr_gt = ingr_gt.to(device)
        quantity_gt = quantity_gt.to(device)

        losses = self.inverse_model(img_inputs, captions = None, target_ingrs = ingr_gt, class_gt = class_gt, sample=False, keep_cnn_gradients=keep_cnn_gradients)

        if not self.train_ingr_only:
            img_embedding = self.inverse_model.image_encoder(img_inputs, keep_cnn_gradients=keep_cnn_gradients)
            
            if self.weighted_loss:
                pred_quantity = self.quantity_model(img_embedding) ## img_embedding = (batch, 512, 49)
                nonzero_mask = (quantity_gt > 0).float()
                example_weights = nonzero_mask.sum(dim=1)
                example_weights = example_weights / (example_weights.mean()+1e-8)

                quantity_loss = self.quantity_criterion(pred_quantity, quantity_gt)
                quantity_loss = (quantity_loss * nonzero_mask).sum(dim=1) / (nonzero_mask.sum(dim=1) + 1e-8)
                quantity_loss = (quantity_loss * example_weights)
            else:
                pred_quantity = self.quantity_model(img_embedding) ## img_embedding = (batch, 512, 49)
                quantity_loss = self.quantity_criterion(pred_quantity, quantity_gt)

            no_quantity = (torch.sum(quantity_gt) == 0) ## if no quantity, assign quantity_loss 0
            quantity_loss[no_quantity] = 0.0

            quantity_loss = quantity_loss.mean()
            losses['quantity_loss'] = quantity_loss
        
        losses['ingr_loss'] = losses['ingr_loss'].mean()
        losses['eos_loss'] = losses['eos_loss'].mean()
        losses['card_penalty'] = losses['card_penalty'].mean()

        return losses

class Quantity_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.global_pool(x)
        # x = x.reshape(x.shape[0], -1)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x

class Quantity_TF(nn.Module):
    def __init__(self, embed_size, num_ingredients, num_layers=3, nhead=8, dropout = 0.3):
        super(Quantity_TF, self).__init__()
        self.num_ingredients = num_ingredients
        self.transformer_layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_size, num_ingredients-1)

    def forward(self, img_features): ## (batch, 512, 49)
        # Use a transformer decoder to predict the quantity for each ingredient
        img_features = img_features.permute(0, 2, 1)  # swap second and third dimensions
        output = img_features.transpose(0, 1)   # (seq_len, batch_size, embed_size)
        for layer in self.transformer_layers:
            output = layer(output, output)  # (seq_len, batch_size, embed_size)
        # output = output.transpose(0, 1)  # (batch_size, seq_len, embed_size)
        output = output[-1]

        # Use a linear layer to map the output of the decoder to the predicted quantities
        output = self.fc(output)

        return output

def get_model(args, ingr_vocab_size, instrs_vocab_size, train_ingr_only=False):

    # build ingredients embedding
    encoder_ingrs = EncoderLabels(args.embed_size, ingr_vocab_size,
                                  args.dropout_encoder, scale_grad=False).to(device)
    # build image model
    if args.ViT:
        encoder_image = EncoderViT(args.embed_size)
    else:
        encoder_image = EncoderCNN(args.embed_size, args.dropout_encoder, args.image_model)
    

    decoder = DecoderTransformer(args.embed_size, instrs_vocab_size,
                                 dropout=args.dropout_decoder_r, seq_length=args.maxseqlen,
                                 num_instrs=args.maxnuminstrs,
                                 attention_nheads=args.n_att, num_layers=args.transf_layers,
                                 normalize_before=True,
                                 normalize_inputs=False,
                                 last_ln=False,
                                 scale_embed_grad=False)

    ingr_decoder = DecoderTransformer(args.embed_size, ingr_vocab_size, dropout=args.dropout_decoder_i,
                                      seq_length=args.maxnumlabels,
                                      num_instrs=1, attention_nheads=args.n_att_ingrs,
                                      pos_embeddings=False,
                                      num_layers=args.transf_layers_ingrs,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)
    # recipe loss
    criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size-1], reduce=False)

    # ingredients loss
    label_loss = nn.BCELoss(reduce=False)
    eos_loss = nn.BCELoss(reduce=False)

    if args.semantic:
        semantic_branch = nn.Linear(args.embed_size, 1048) ## numClasses = 1048
    else:
        semantic_branch = None

    inverse_model = InverseCookingModel(encoder_ingrs, decoder, ingr_decoder, encoder_image, semantic_branch, ## semantic_branch
                                crit=criterion, crit_ingr=label_loss, crit_eos=eos_loss,
                                pad_value=ingr_vocab_size-1,
                                ingrs_only=args.ingrs_only, recipe_only=args.recipe_only,
                                label_smoothing=args.label_smoothing_ingr)

    #### 밑에 추가
    quantity_model = Quantity_TF(embed_size = args.embed_size, num_ingredients = ingr_vocab_size, num_layers=args.transf_layers_quantity)
    # quantity_model = Quantity_MLP(input_size = args.embed_size, hidden_size = 1024, output_size = ingr_vocab_size-1)

    model = Inverse_Quantity(inverse_model, quantity_model, args.quantity_only, train_ingr_only, args.weighted_loss)
    ####

    return model


class InverseCookingModel(nn.Module):
    def __init__(self, ingredient_encoder, recipe_decoder, ingr_decoder, image_encoder, semantic_branch,
                 crit=None, crit_ingr=None, crit_eos=None,
                 pad_value=0, ingrs_only=True,
                 recipe_only=False, label_smoothing=0.0):

        super(InverseCookingModel, self).__init__()

        self.ingredient_encoder = ingredient_encoder
        self.recipe_decoder = recipe_decoder
        self.image_encoder = image_encoder
        self.ingredient_decoder = ingr_decoder
        self.crit = crit
        self.crit_ingr = crit_ingr
        self.pad_value = pad_value
        self.ingrs_only = ingrs_only
        self.recipe_only = recipe_only
        self.crit_eos = crit_eos
        self.label_smoothing = label_smoothing
        self.semantic_branch = semantic_branch ## None or linear
        if self.semantic_branch is not None:
            weights_class = torch.Tensor(1048).fill_(1) ## numClasses = 1048
            weights_class[0] = 0 # the background class is set to 0, i.e. ignore
            # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
            self.semantic_crit = nn.CrossEntropyLoss(weight=weights_class).to(device)
            self.pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, img_inputs, captions, target_ingrs, class_gt=None,
                sample=False, keep_cnn_gradients=False):

        if sample:
            return self.sample(img_inputs, greedy=True)

        if captions is not None:
            targets = captions[:, 1:]
            targets = targets.contiguous().view(-1)

        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)

        losses = {}

        ## semantic prediction #####
        if self.semantic_branch is not None:
            pool = self.pool(img_features).squeeze(-1)
            class_prediction = self.semantic_branch(pool)
            semantic_loss = self.semantic_crit(class_prediction, class_gt.squeeze(-1))
            if torch.isnan(semantic_loss) and torch.sum(class_gt) == 0:
                print("semantic loss error (all class_gt 0)") ## class_gt 다 0이기는 한데..
                semantic_loss = torch.tensor(0.0).to(device) ## all class_gt = 0...
            losses['semantic_loss'] = semantic_loss
        
        target_one_hot = label2onehot(target_ingrs, self.pad_value)
        target_one_hot_smooth = label2onehot(target_ingrs, self.pad_value)

        # ingredient prediction
        if not self.recipe_only:
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

        if self.ingrs_only:
            return losses

        # encode ingredients
        target_ingr_feats = self.ingredient_encoder(target_ingrs)
        target_ingr_mask = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)

        target_ingr_mask = target_ingr_mask.float().unsqueeze(1)

        outputs, ids = self.recipe_decoder(target_ingr_feats, target_ingr_mask, captions, img_features)

        outputs = outputs[:, :-1, :].contiguous()
        outputs = outputs.view(outputs.size(0) * outputs.size(1), -1)

        loss = self.crit(outputs, targets)

        losses['recipe_loss'] = loss

        return losses

    def sample(self, img_inputs, greedy=True, temperature=1.0, beam=-1, true_ingrs=None):

        outputs = dict()

        img_features = self.image_encoder(img_inputs)

        if not self.recipe_only:
            ingr_ids, ingr_probs = self.ingredient_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                                  beam=-1,
                                                                  img_features=img_features, first_token_value=0,
                                                                  replacement=False)

            # mask ingredients after finding eos
            sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
            ingr_ids[sample_mask == 0] = self.pad_value

            outputs['ingr_ids'] = ingr_ids
            outputs['ingr_probs'] = ingr_probs.data

            mask = sample_mask
            input_mask = mask.float().unsqueeze(1)
            input_feats = self.ingredient_encoder(ingr_ids)

        if self.ingrs_only:
            return outputs

        # option during sampling to use the real ingredients and not the predicted ones to infer the recipe
        if true_ingrs is not None:
            input_mask = mask_from_eos(true_ingrs, eos_value=0, mult_before=False)
            true_ingrs[input_mask == 0] = self.pad_value
            input_feats = self.ingredient_encoder(true_ingrs)
            input_mask = input_mask.unsqueeze(1)

        ids, probs = self.recipe_decoder.sample(input_feats, input_mask, greedy, temperature, beam, img_features, 0,
                                                last_token_value=1)

        outputs['recipe_probs'] = probs.data
        outputs['recipe_ids'] = ids

        return outputs
