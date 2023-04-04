# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import random
import numpy as np
from vit_pytorch import ViT
import timm

class EncoderViT(nn.Module):
    def __init__(self, embed_size):
        super(EncoderViT, self).__init__()

        # self.vit = ViT(image_size = 224, patch_size = 16, num_classes = 1000, dim = 768, depth=12, heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit = nn.Sequential(*list(self.vit.children())[:-1])
        self.linear = nn.Linear(768, embed_size)
    
    def forward(self, images, keep_cnn_gradients = False):
        if keep_cnn_gradients:
            raw_conv_feats = self.vit(images)
        else:
            with torch.no_grad():
                raw_conv_feats = self.vit(images)
        features = self.linear(raw_conv_feats) ## raw_conv_feats = (batch, 196, 768)
        features = features.permute(0, 2, 1) ## feature = (batch, 196, 512) -> (batch, 512, 196)

        return features

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # resnet = globals()['image_model'](pretrained=pretrained) ## 원래 이거였음
        resnet = resnet50(pretrained = pretrained)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Sequential(nn.Conv2d(resnet.fc.in_features, embed_size, kernel_size=1, padding=0),
                                    nn.Dropout2d(dropout)) ## embed_size = 512

    def forward(self, images, keep_cnn_gradients=False):
        """Extract feature vectors from input images."""

        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            with torch.no_grad():
                raw_conv_feats = self.resnet(images) ## (batch, 2048, 7, 7)
        features = self.linear(raw_conv_feats) ## feaetures = (batch_size, embed_size = 512, 7, 7)
        features = features.view(features.size(0), features.size(1), -1) ## (batch_size, embed_size, 7*7 = 49)

        return features


class EncoderLabels(nn.Module):
    def __init__(self, embed_size, num_classes, dropout=0.5, embed_weights=None, scale_grad=False):

        super(EncoderLabels, self).__init__()
        embeddinglayer = nn.Embedding(num_classes, embed_size, padding_idx=num_classes-1, scale_grad_by_freq=scale_grad)
        if embed_weights is not None:
            embeddinglayer.weight.data.copy_(embed_weights)
        self.pad_value = num_classes - 1
        self.linear = embeddinglayer
        self.dropout = dropout
        self.embed_size = embed_size

    def forward(self, x, onehot_flag=False):

        if onehot_flag:
            embeddings = torch.matmul(x, self.linear.weight)
        else:
            embeddings = self.linear(x)

        embeddings = nn.functional.dropout(embeddings, p=self.dropout, training=self.training)
        embeddings = embeddings.permute(0, 2, 1).contiguous()

        return embeddings
