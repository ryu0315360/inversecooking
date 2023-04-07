# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import random
import json
import lmdb


class Recipe1MDataset(data.Dataset):

    def __init__(self, data_dir, aux_data_dir, split, maxseqlen, maxnuminstrs, maxnumlabels, maxnumims, extended_1M,
                 transform=None, max_num_samples=-1, use_lmdb=False, suff=''):

        self.ingrs_vocab = pickle.load(open('/home/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb'))
        self.instrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_vocab_toks.pkl'), 'rb'))

        if extended_1M:
            if split == 'val_1M':
                self.dataset = pickle.load(open(os.path.join('/home/donghee/inversecooking/data/recipe1m_'+split+'.pkl'), 'rb'))
            else:
                self.dataset = pickle.load(open(os.path.join('/home/donghee/inversecooking/data/download_recipe1m_'+split+'.pkl'), 'rb'))## weight_recipe1m
        
        else: ## original 1M data
            self.dataset = pickle.load(open(os.path.join('/home/donghee/inversecooking/data/recipe1m_'+split+'.pkl'), 'rb'))

        self.label2word = self.get_ingrs_vocab()

        self.extended_1M = extended_1M
        self.use_lmdb = use_lmdb
        if use_lmdb:
            if split == 'val_1M+':
                self.image_file = lmdb.open(os.path.join(aux_data_dir, 'lmdb_' + 'val_1M'), max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)
                self.extended_image_file = lmdb.open(os.path.join(aux_data_dir, 'download_lmdb_' + split), max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)
            elif split == 'val_1M':
                self.image_file = lmdb.open(os.path.join(aux_data_dir, 'lmdb_' + split), max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)
                self.extended_image_file = self.image_file
            else: ## train_1M+, train_1M,
                self.image_file = lmdb.open(os.path.join(aux_data_dir, 'lmdb_' + split), max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)
                self.extended_image_file = lmdb.open(os.path.join(aux_data_dir, 'download_lmdb_' + split), max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)
                                    
        with open('/home/donghee/inversecooking/layer2_ids.json', 'r') as f:
            self.layer2_ids = set(json.load(f))
        self.ids = []
        self.split = split
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue ## image 있는 것만 취함
            self.ids.append(i)

        self.root = os.path.join(data_dir, 'images', split)
        self.transform = transform
        self.max_num_labels = maxnumlabels
        self.maxseqlen = maxseqlen
        self.max_num_instrs = maxnuminstrs
        self.maxseqlen = maxseqlen*maxnuminstrs
        self.maxnumims = maxnumims
        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples] ## 4096만 가지고 val 하는 거!! 그래서 매 epoch 하는 거였어.. 대박..

    def get_instrs_vocab(self):
        return self.instrs_vocab

    def get_instrs_vocab_size(self):
        return len(self.instrs_vocab)

    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient

    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.dataset[self.ids[index]]
        img_id = sample['id']
        # captions = sample['tokenized']
        paths = sample['images'][0:self.maxnumims] ## 이 path가 진짜 img_id네!

        idx = index

        labels = sample['ingredients']
        title = sample['title']
        

        # tokens = []
        # tokens.extend(title)
        # # add fake token to separate title from recipe
        # tokens.append('<eoi>')
        # for c in captions:
        #     tokens.extend(c)
        #     tokens.append('<eoi>')

        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        pos = 0

        true_ingr_idxs = []
        for i in range(len(labels)):
            true_ingr_idxs.append(self.ingrs_vocab(labels[i]))

        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_idx = self.ingrs_vocab(label) ## return integer id
            if label_idx not in ilabels_gt:
                ilabels_gt[pos] = label_idx
                pos += 1

        if pos == self.max_num_labels:
            ilabels_gt[-1] = self.ingrs_vocab('<end>') ## 마지막 ingr 그냥 날려ㅠ
        else:
            ilabels_gt[pos] = self.ingrs_vocab('<end>') ## 0
        ingrs_gt = torch.from_numpy(ilabels_gt).long()

        ##### quantity part
        try:
            quantity = sample['quantity']
            unit = sample['unit']
            quantity_gt = np.zeros(len(self.ingrs_vocab)-1) ## except <pad> ## 1487
            unit_gt = np.zeros(len(self.ingrs_vocab)-1)
            exist_quantity = False

            if len(quantity) != 0 and len(labels) == pos: ## if there's quantity data and all the ingredients are valid, unique ## 모든 ingr가 제대로 매핑
                exist_quantity = True
                i = 0
                while ilabels_gt[i] != 0: ## until end
                    quantity_gt[int(ilabels_gt[i])] = quantity[i]
                    # unit_gt[int(ilabels_gt[i])] = unit[i] ## could not convert string to float
                    i += 1
                # print("CAN utilize quantity")
            elif len(quantity) != 0 and len(labels) != pos:
                j=0

            quantity_gt = torch.from_numpy(quantity_gt).float()
        
        except: ## no quantity (val_1M)
            quantity_gt = np.zeros(len(self.ingrs_vocab)-1)
            quantity_gt = torch.from_numpy(quantity_gt).float()
        
        ##### TODO unit 어떻게 return 할 건지..
                


        if len(paths) == 0:
            path = None
            image_input = torch.zeros((3, 224, 224)) ## 
            print("*** No image")
        else:
            if self.split == 'train':
                img_idx = np.random.randint(0, len(paths))
            else:
                img_idx = 0
            path = paths[img_idx]
            if self.use_lmdb:
                if sample['id'] in self.layer2_ids:
                    lmdb_file = self.image_file
                else:
                    lmdb_file = self.extended_image_file
                try:
                    with lmdb_file.begin(write=False) as txn:
                        image = txn.get(path.encode())
                        image = np.fromstring(image, dtype=np.uint8)
                        image = np.reshape(image, (256, 256, 3)) ## TODO 여기서 shape 안 맞아
                    image = Image.fromarray(image.astype('uint8'), 'RGB')
                except:
                    # print ("Image id not found in lmdb. Loading jpeg file...")
                    # image = Image.open(os.path.join(self.root, path[0], path[1],
                    #                                     path[2], path[3], path)).convert('RGB')
                    image_input = None
                    print("Fail to load image. not use this data.")
                    return image_input, ingrs_gt, quantity_gt, img_id, path
            else:
                image = Image.open(os.path.join(self.root, path[0], path[1], path[2], path[3], path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            image_input = image

        ## class
        class_gt = np.array([sample['classes']])
        class_gt = torch.from_numpy(class_gt).long()
        ####

        # Convert caption (string) to word ids. ## instructions
        # caption = []

        # caption = self.caption_to_idxs(tokens, caption)
        # caption.append(self.instrs_vocab('<end>'))

        # caption = caption[0:self.maxseqlen]
        # target = torch.Tensor(caption)

        return image_input, ingrs_gt, quantity_gt, class_gt, img_id, path

    def __len__(self):
        return len(self.ids)

    # def caption_to_idxs(self, tokens, caption):

    #     caption.append(self.instrs_vocab('<start>'))
    #     for token in tokens:
    #         caption.append(self.instrs_vocab(token))
    #     return caption


def collate_fn_quantity(data): ## quantity 데이터 있는 경우만 사용...

    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    # image_input, captions, ingrs_gt, img_id, path, pad_value = zip(*data)
    data = [sample for sample in data if torch.sum(sample[2]) != 0 and sample[0] != None] ## 여기서 valid 하지 않은 경우는.. quantity data가 들어왔지만, ingredient로 맵핑하는 과정에서 정확히 다 매핑이 안 돼서 버려지는 거ㅠ

    if not data:
        return None
    
    image_input, ingrs_gt, quantity_gt, class_gt, img_id, path = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)
    quantity_gt = torch.stack(quantity_gt,0) ## Q) 이거 왜 하는거지ㅠ
    class_gt = torch.concat(class_gt)
    # title = torch.stack(title, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(cap) for cap in captions]
    # targets = torch.ones(len(captions), max(lengths)).long()*pad_value[0]

    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]

    return image_input, ingrs_gt, quantity_gt, class_gt, img_id, path

def collate_fn_valid_image(data):

    data = [sample for sample in data if sample[0] is not None] ## image 가 None이 아닌 경우만 (jpeg, lmdb 둘 중 하나는 성공한 경우)

    if not data:
        return None
    
    image_input, ingrs_gt, quantity_gt, class_gt, img_id, path = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)
    quantity_gt = torch.stack(quantity_gt,0) ## Q) 이거 왜 하는거지ㅠ
    class_gt = torch.concat(class_gt)

    return image_input, ingrs_gt, quantity_gt, class_gt, img_id, path


def get_loader(data_dir, aux_data_dir, split, maxseqlen,
               maxnuminstrs, maxnumlabels, maxnumims, transform, batch_size, extended_1M,
               shuffle, num_workers, drop_last=False,
               max_num_samples=-1,
               use_lmdb=False,
               suff=''):

    dataset = Recipe1MDataset(data_dir=data_dir, aux_data_dir=aux_data_dir, split=split,
                              maxseqlen=maxseqlen, maxnumlabels=maxnumlabels, maxnuminstrs=maxnuminstrs,
                              maxnumims=maxnumims, extended_1M=extended_1M,
                              transform=transform,
                              max_num_samples=max_num_samples,
                              use_lmdb=use_lmdb,
                              suff=suff)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_fn_valid_image, pin_memory=True) # collate_fn=collate_fn_quantity - quantity 있는 데이터만 이용하고 싶을 때
    return data_loader, dataset
