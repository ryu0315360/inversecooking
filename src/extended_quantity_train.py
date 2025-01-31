from args import get_parser
import torch
import torch.nn as nn
import torchvision.models as models
from einops.layers.torch import Rearrange
import torch.autograd as autograd
import numpy as np
import os
import random
import pickle
from inversecooking.src.my_data_loader import get_loader, Recipe1MDataset
from build_vocab import Vocabulary
from my_model import get_model
from torchvision import transforms
import sys
import json
import time
import torch.backends.cudnn as cudnn
from utils.tb_visualizer import Visualizer
from my_model import mask_from_eos, label2onehot
from utils.metrics import softIoU, compute_metrics, update_error_types
import random
from tqdm import tqdm
import argparse
from utils.output_utils import get_ingrs
import logging
import sys
# sys.path.append('/home/donghee/CLIP')
# import clip

# from temp import quantity_counter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

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

# class Quantity_TF(nn.Module):
#     def __init__(self, num_classes):
#         super(Quantity_TF, self).__init__()
#         self.backbone = models.vit_base_patch16_224(pretrained=True)
#         self.rearrange = Rearrange('b e h w -> b (h w) e')
#         self.fc = nn.linear(512, num_classes)
    
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.rearrange(x)
#         x = self.fc(x)
#         return x

class Quantity_TF(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3):
        super(Quantity_TF, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden_size, nhead=8) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        return x.squeeze()

class Inverse_Quantity(nn.Module):
    def __init__(self, inverse_model, quantity_model):
        super().__init__()

        self.inverse_model = inverse_model
        self.quantity_model = quantity_model
        self.quantity_criterion = nn.MSELoss(reduce = False)
        self.quantity_criterion = self.quantity_criterion.to(device)

    def forward(self, img_inputs, ingr_gt, quantity_gt, sample = False, val=False):

        if sample:
            return self.inverse_model.sample(img_inputs, greedy=True) ## return outputs

        img_inputs = img_inputs.to(device)
        ingr_gt = ingr_gt.to(device)
        quantity_gt = quantity_gt.to(device)

        # losses = self.inverse_model(img_inputs, captions = None, target_ingrs = ingr_gt)
     
        with torch.no_grad():
            losses = self.inverse_model(img_inputs, captions = None, target_ingrs = ingr_gt)
            img_embedding = self.inverse_model.image_encoder(img_inputs, keep_cnn_gradients=False)
        
        pred_quantity = self.quantity_model(img_embedding) ## img_embedding = (batch, 512, 49)
        nonzero_mask = (quantity_gt > 0).float()
        example_weights = nonzero_mask.sum(dim=1)
        example_weights = example_weights / (example_weights.mean()+1e-8)

        quantity_loss = self.quantity_criterion(pred_quantity, quantity_gt)
        quantity_loss = (quantity_loss * nonzero_mask).sum(dim=1) / (nonzero_mask.sum(dim=1) + 1e-8)

        quantity_loss = (quantity_loss * example_weights)
        quantity_loss = quantity_loss.mean()
        losses['quantity_loss'] = quantity_loss
        losses['ingr_loss'] = losses['ingr_loss'].mean()
        losses['eos_loss'] = losses['eos_loss'].mean()
        losses['card_penalty'] = losses['card_penalty'].mean()

        if val == True:
            return losses, pred_quantity

        return losses
    
def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr']*decay_factor

def save_model(model, optimizer, checkpoints_dir, suff=''):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(
            checkpoints_dir, 'model' + suff + '.ckpt'))

    else:
        torch.save(model.state_dict(), os.path.join(
            checkpoints_dir, 'model' + suff + '.ckpt'))

    torch.save(optimizer.state_dict(), os.path.join(
        checkpoints_dir, 'optim' + suff + '.ckpt'))


def main(args):
    #######
    mode = 'train'
    save_dir = '/home/donghee/inversecooking/results'
    project_name = 'quantity_only_tf'
    model_name = 'from_best'
    train_ingr_only = False
    train_quantity_only = True
    resume = True
    resume_model_path = '/home/donghee/inversecooking/data/modelbest.ckpt'
    cnn_train = False
    # resume_optim_path = '/home/donghee/inversecooking/results/train_ingr_only/from_best/checkpoints/optim.ckpt'
    #######

    # Create model directory & other aux folders for logging
    where_to_save = os.path.join(save_dir, project_name, model_name) ## results/project_name/model_name
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    logs_dir = os.path.join(where_to_save, 'logs')
    tb_logs = os.path.join(save_dir, project_name, 'tb_logs', model_name)
    make_dir(where_to_save)
    make_dir(logs_dir)
    make_dir(checkpoints_dir)
    make_dir(tb_logs)

    logging.basicConfig(filename = os.path.join(where_to_save, 'my_log.log'), level = logging.INFO)

    if args.tensorboard:
        logger = Visualizer(tb_logs, name='visual_results')
    
    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(logs_dir, 'train.log'))
        sys.stdout = open(os.path.join(logs_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'train.err'), 'w')
    
    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))

    # patience init
    curr_pat = 0


    data_loaders = {}
    datasets = {}

    data_dir = args.recipe1m_dir

    for split in ['train', 'val', 'test']:

        transforms_list = [transforms.Resize((args.image_size))]

        if split == 'train':
            # Image preprocessing, normalization for the pretrained resnet
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            transforms_list.append(transforms.RandomCrop(args.crop_size))

        else:
            transforms_list.append(transforms.CenterCrop(args.crop_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))

        transform = transforms.Compose(transforms_list)
        max_num_samples = max(args.max_eval, args.batch_size) if split == 'val' else -1
        if mode == 'clip_test':
            max_num_samples = -1
        data_loaders[split], datasets[split] = get_loader(data_dir, args.aux_data_dir, split,
                                                          args.maxseqlen,
                                                          args.maxnuminstrs,
                                                          args.maxnumlabels,
                                                          args.maxnumims,
                                                          transform, args.batch_size,
                                                          shuffle=split == 'train', num_workers=args.num_workers,
                                                          drop_last=True,
                                                          max_num_samples=max_num_samples,
                                                          use_lmdb=True,
                                                          suff='')
    
    data_loaders['val_origin'], datasets['val_origin'] = get_loader(data_dir, args.aux_data_dir, 'val_origin',
                                                          args.maxseqlen,
                                                          args.maxnuminstrs,
                                                          args.maxnumlabels,
                                                          args.maxnumims,
                                                          transform, args.batch_size,
                                                          shuffle='val_origin' == 'train', num_workers=args.num_workers,
                                                          drop_last=True,
                                                          max_num_samples=max_num_samples,
                                                          use_lmdb=True,
                                                          suff='')

    # quantity_counter(data_loaders)
    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()

    idx2ingr = datasets[split].ingrs_vocab.idx2word

    if train_ingr_only:
        model = get_model(args, ingr_vocab_size, 23231)
        keep_cnn_gradients = False

        if resume:
            model.load_state_dict(torch.load(resume_model_path, map_location=map_loc))
        
        params = list(model.ingredient_decoder.parameters())
        print ("ingredient decoder params:", sum(p.numel() for p in params if p.requires_grad))
        params_linear = list(model.image_encoder.linear.parameters())
        params += params_linear
        print ("image linear params:", sum(p.numel() for p in params_linear if p.requires_grad))
        params_cnn = list(model.image_encoder.resnet.parameters())
        print ("CNN params:", sum(p.numel() for p in params_cnn if p.requires_grad))

        if cnn_train:
            optimizer = torch.optim.Adam([{'params': params}, {'params': params_cnn,
                                                            'lr': args.learning_rate*args.scale_learning_rate_cnn}],
                                        lr=args.learning_rate, weight_decay=args.weight_decay)
            keep_cnn_gradients = True
            print ("Fine tuning resnet")
        else:
            optimizer = torch.optim.Adam(params, lr=args.learning_rate)
            print("Train only linear layer")
        
        ## TODO
        ## from_best 이면 lr 좀 더 작게 해야하지 않을까..

    else:
        # quantity_model = Quantity_MLP(input_size = args.embed_size, hidden_size = 1024, output_size = ingr_vocab_size-1)
        quantity_model = Quantity_TF(input_size=512*49, hidden_size=512, num_classes=1488, num_layers=3) ## TODO hardcode here
        inverse_model = get_model(args, ingr_vocab_size, 23231) ## TODO hardcode here
        keep_cnn_gradients = False

        if resume:
            inverse_model.load_state_dict(torch.load(resume_model_path, map_location=map_loc))

        for p in inverse_model.parameters():
            p.requires_grad = False

        # for p in inverse_model.image_encoder.linear.parameters():
        #     p.requires_grad = True
        
        # for p in inverse_model.ingredient_decoder.parameters():
        #     p.requires_grad = True
        
        if train_quantity_only:
            model = quantity_model
            params = list(model.parameters())
        else:
            model = Inverse_Quantity(inverse_model, quantity_model)
            decay_factor = 1.0
            params = list(model.quantity_model.parameters())

        # params = list(inverse_model.ingredient_decoder.parameters())
        # params = list(model.quantity_model.parameters())
        # params_linear = list(model.inverse_model.image_encoder.linear.parameters())
        # params += params_quantity
        # params += params_linear

        print ("quantity decoder params:", sum(p.numel() for p in params if p.requires_grad))
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # if resume:
    #     optimizer.load_state_dict(torch.load(resume_optim_path, map_location=map_loc))
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(device)
    #     model.load_state_dict(torch.load(resume_model_path, map_location=map_loc))

    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    cudnn.benchmark = True

    if not hasattr(args, 'current_epoch'):
        args.current_epoch = 0

    train_losses = []
    val_losses = []
    best_val = float('inf')
    es_best = best_val ##
    start = args.current_epoch
    if mode == 'train':
        for epoch in tqdm(range(start, args.num_epochs)):
            print("======== EPOCH {} =========".format(epoch))
            logging.info(f'====== EPOCH : {epoch} =======')

            if args.tensorboard:
                logger.reset()

            if args.decay_lr:
                frac = epoch // args.lr_decay_every
                decay_factor = args.lr_decay_rate ** frac ## 0.99
                new_lr = args.learning_rate*decay_factor
                print ('Epoch %d. lr: %.5f'%(epoch, new_lr)) ## no decay..
                set_lr(optimizer, decay_factor)
            
            train_loss = 0.0
            n_train = 0

            total_loss_dict = {'quantity_loss': [], 'ingr_loss': [],
                               'eos_loss': [], 'loss': [],
                               'iou': [], 'perplexity': [], 'iou_sample': [],
                               'f1': [],
                               'card_penalty': []}

            torch.cuda.synchronize()
            start = time.time()

            for i, batch in enumerate(data_loaders['train']):
                model.train()
                split = 'train'

                if batch is None:
                    continue
                
                n_train += batch[0].shape[0]
                img_inputs, ingr_gt, quantity_gt, recipe_ids, img_ids = batch
                img_inputs = img_inputs.to(device)
                ingr_gt = ingr_gt.to(device)
                quantity_gt = quantity_gt.to(device)
                
                if not train_ingr_only: ## quantity도 train
                    losses = model.forward(img_inputs, ingr_gt, quantity_gt)
                else:
                    captions = None
                    losses = model.forward(img_inputs, captions, ingr_gt)

                # loss = losses['quantity_loss']
                ## TODO - ingr_true / quantity loss scale..

                ## 4GPUs
                ingr_loss = losses['ingr_loss']
                ingr_loss = ingr_loss.mean()
                losses['ingr_loss'] = ingr_loss.item()

                eos_loss = losses['eos_loss']
                eos_loss = eos_loss.mean()
                losses['eos_loss'] = eos_loss.item()

                iou_seq = losses['iou']
                iou_seq = iou_seq.mean()
                losses['iou'] = iou_seq.item()

                card_penalty = losses['card_penalty'].mean()
                losses['card_penalty'] = card_penalty.item()

                if not train_ingr_only: ## quantity also train
                    quantity_loss = losses['quantity_loss']
                    quantity_loss = quantity_loss.mean()
                    losses['quantity_loss'] = quantity_loss.item()
                    
                    loss = 1000.0*ingr_loss + 0.01*quantity_loss + 1.0*eos_loss + 1.0*card_penalty
                    logging.info(f'** total loss : {loss.item()}, ingr_loss = {ingr_loss.item()}, quantity_loss ={quantity_loss.item()}, eos_loss = {eos_loss.item()}, card_penalty = {card_penalty.item()}')
                
                else: ## only ingr_true train
                    quantity_loss = 0.0
                    loss = 1000.0*ingr_loss + 1.0*eos_loss + 1.0*card_penalty
                    logging.info(f'** total loss : {loss.item()}, ingr_loss = {ingr_loss.item()}, eos_loss = {eos_loss.item()}, card_penalty = {card_penalty.item()}')

                losses['loss'] = loss.item()

                model.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                for key in losses.keys():
                    total_loss_dict[key].append(losses[key])

                # Print log info
                if args.log_step != -1 and i % args.log_step == 0:
                    elapsed_time = time.time()-start
                    lossesstr = ""
                    for k in total_loss_dict.keys():
                        if len(total_loss_dict[k]) == 0:
                            continue
                        this_one = "%s: %.8f" % (k, np.mean(total_loss_dict[k][-args.log_step:]))
                        lossesstr += this_one + ', '
                    # this only displays nll loss on captions, the rest of losses will be in tensorboard logs
                    strtoprint = 'Split: %s, Epoch [%d/%d], Step [%d/%d], Losses: %sTime: %.4f' % (split, epoch,
                                                                                                   args.num_epochs, i,
                                                                                                   len(data_loaders[split]),
                                                                                                   lossesstr,
                                                                                                   elapsed_time)
                    print(strtoprint)

                    if args.tensorboard:
                        # logger.histo_summary(model=model, step=total_step * epoch + i)
                        logger.scalar_summary(mode=split+'_iter', epoch=len(data_loaders[split])*epoch+i,
                                              **{k: np.mean(v[-args.log_step:]) for k, v in total_loss_dict.items() if v})

                    torch.cuda.synchronize()
                    start = time.time()
                del loss, losses, img_inputs
                    
            
            train_loss = train_loss / len(data_loaders['train'])
            train_losses.append(train_loss)
            if epoch == 0:
                print("# training data points: ", n_train)
            print("Losses : ",train_losses)

            if args.tensorboard:
                # 1. Log scalar values (scalar summary)
                logger.scalar_summary(mode=split,
                                      epoch=epoch,
                                      **{k: np.mean(v) for k, v in total_loss_dict.items() if v})

            
            n_val = 0
            total_loss_dict = {'quantity_loss': [], 'ingr_loss': [],
                               'eos_loss': [], 'loss': [],
                               'iou': [], 'perplexity': [], 'iou_sample': [],
                               'f1': [],
                               'card_penalty': []}
            
            error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0,
                           'tp_all': 0, 'fp_all': 0, 'fn_all': 0}
            
            torch.cuda.synchronize()
            start = time.time()

            if (epoch+1)%args.val_freq == 0 or epoch == 0:
                logging.info(f'===== VAL (EPOCH: {epoch}) =====')
                model.eval()
                # split = 'val'

                with torch.no_grad():
                    val_loss = 0.0
                    splits = ['val_origin', 'val']
                    for split in splits:
                        for i, batch in enumerate(data_loaders[split]):
                            
                            if batch is None:
                                continue
                            
                            n_val += batch[0].shape[0]
                            img_inputs, ingr_gt, quantity_gt, recipe_ids, img_ids = batch
                            img_inputs = img_inputs.to(device)
                            ingr_gt = ingr_gt.to(device)
                            quantity_gt = quantity_gt.to(device)

                            if not train_ingr_only: ## quantity도 train
                                losses, pred_quantity = model.forward(img_inputs, ingr_gt, quantity_gt, val = True)
                            else:
                                captions = None
                                losses = model.forward(img_inputs, captions, ingr_gt)
                            
                            ## 4GPUs
                            ingr_loss = losses['ingr_loss']
                            ingr_loss = ingr_loss.mean()
                            losses['ingr_loss'] = ingr_loss.item()

                            eos_loss = losses['eos_loss']
                            eos_loss = eos_loss.mean()
                            losses['eos_loss'] = eos_loss.item()

                            iou_seq = losses['iou']
                            iou_seq = iou_seq.mean()
                            losses['iou'] = iou_seq.item()

                            card_penalty = losses['card_penalty'].mean()
                            losses['card_penalty'] = card_penalty.item()

                            if not train_ingr_only: ## quantity also train
                                quantity_loss = losses['quantity_loss']
                                quantity_loss = quantity_loss.mean()
                                losses['quantity_loss'] = quantity_loss.item()
                                
                                loss = 1000.0*ingr_loss + 0.01*quantity_loss + 1.0*eos_loss + 1.0*card_penalty
                                logging.info(f'** {split}- total loss : {loss.item()}, ingr_loss = {ingr_loss.item()}, quantity_loss ={quantity_loss.item()}, eos_loss = {eos_loss.item()}, card_penalty = {card_penalty.item()}')
                            
                            else: ## only ingr_true train
                                quantity_loss = 0.0
                                loss = 1000.0*ingr_loss + 1.0*eos_loss + 1.0*card_penalty
                                logging.info(f'** {split}- total loss : {loss.item()}, ingr_loss = {ingr_loss.item()}, eos_loss = {eos_loss.item()}, card_penalty = {card_penalty.item()}')

                            losses['loss'] = loss.item()

                            val_loss += loss.item()

                            ## sample
                            if not train_ingr_only: ## quantity also train
                                outputs = model.forward(img_inputs, ingr_gt, quantity_gt, sample = True)
                            else: ## only ingr
                                outputs = model.forward(img_inputs, captions, ingr_gt, sample = True)

                            ingr_ids_greedy = outputs['ingr_ids']

                            mask = mask_from_eos(ingr_ids_greedy, eos_value=0, mult_before=False)
                            ingr_ids_greedy[mask == 0] = ingr_vocab_size-1
                            pred_one_hot = label2onehot(ingr_ids_greedy, ingr_vocab_size-1)
                            target_one_hot = label2onehot(ingr_gt, ingr_vocab_size-1)
                            iou_sample = softIoU(pred_one_hot, target_one_hot)
                            iou_sample = iou_sample.sum() / (torch.nonzero(iou_sample.data).size(0) + 1e-6)
                            losses['iou_sample'] = iou_sample.item()

                            update_error_types(error_types, pred_one_hot, target_one_hot)

                            pred_ingrs = ingr_ids_greedy

                            del outputs, pred_one_hot, target_one_hot, iou_sample

                            
                            if random.random() < 0.05: ## of np.random.uniform()
                                if not train_ingr_only: ## quantity also
                                    check_example(logging, idx2ingr, ingr_gt[0], quantity_gt[0], pred_quantity[0], recipe_ids[0], pred_ingrs[0])
                                else: ## only ingr
                                    check_example(logging=logging, idx2ingr=idx2ingr, ingr_gt=ingr_gt[0], quantity_gt = None, pred_quantity = None, img_id = recipe_ids[0], pred_ingr = pred_ingrs[0])
                            for key in losses.keys():
                                total_loss_dict[key].append(losses[key])

                            # Print log info
                            if args.log_step != -1 and i % args.log_step == 0:
                                elapsed_time = time.time()-start
                                lossesstr = ""
                                for k in total_loss_dict.keys():
                                    if len(total_loss_dict[k]) == 0:
                                        continue
                                    this_one = "%s: %.8f" % (k, np.mean(total_loss_dict[k][-args.log_step:]))
                                    lossesstr += this_one + ', '
                                # this only displays nll loss on captions, the rest of losses will be in tensorboard logs
                                strtoprint = 'Split: %s, Epoch [%d/%d], Step [%d/%d], Losses: %sTime: %.4f, iou: %.4f' % (split, epoch,
                                                                                                            args.num_epochs, i,
                                                                                                            len(data_loaders[split]),
                                                                                                            lossesstr,
                                                                                                            elapsed_time, losses['iou'])
                                print(strtoprint)

                                if args.tensorboard:
                                    # logger.histo_summary(model=model, step=total_step * epoch + i)
                                    logger.scalar_summary(mode=split+'_iter', epoch=len(data_loaders[split])*epoch+i,
                                                        **{k: np.mean(v[-args.log_step:]) for k, v in total_loss_dict.items() if v})

                                torch.cuda.synchronize()
                                start = time.time()
                            del loss, losses, img_inputs

                            ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'f1_ingredients': [], 'dice': []}
                            compute_metrics(ret_metrics, error_types,
                                            ['accuracy', 'f1', 'jaccard', 'f1_ingredients', 'dice'], eps=1e-10,
                                            weights=None)
                            total_loss_dict['f1'] = ret_metrics['f1']

                        if epoch == 0:
                            print("# val data points: ", n_val)
                        val_loss = val_loss / len(data_loaders[split])
                        val_losses.append(val_loss)
                        print("Val losses: ", val_losses)

                        is_best = val_loss < best_val
                        best_val = min(val_loss, best_val)
                        print("* current val loss: ", val_loss)
                        print("* best val_loss: ", best_val)

                es_value = np.mean(total_loss_dict[args.es_metric])

                # save current model as well
                save_model(model, optimizer, checkpoints_dir, suff='')
                if (args.es_metric == 'loss' and es_value < es_best) or (args.es_metric == 'iou_sample' and es_value > es_best):
                    es_best = es_value
                    save_model(model, optimizer, checkpoints_dir, suff='best')
                    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))
                    curr_pat = 0
                    print('Saved checkpoint.')
                else:
                    curr_pat += 1

                if curr_pat > args.patience:
                    break

        if args.tensorboard:
            logger.close()   

    elif mode == 'clip_test':
        model, preprocess = clip.load("ViT-B/32", device=device)
        text = clip.tokenize(['A photo of a food', 'A photo of non-food']).to(device)

        paths = []
        threshold = 0.85
        food_probs = []
        with torch.no_grad():
            for split in ['val', 'train', 'test']:
                for i, batch in enumerate(tqdm(data_loaders[split])):
                    img_inputs, ingr_gt, quantity_gt, recipe_ids, img_ids = batch
                    img_inputs = img_inputs.to(device)
                    text = text.to(device)
                    logits_per_image, logits_per_text, image_embedding, text_embedding = model(img_inputs, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                    # probs = np.array(probs)[:,0]
                    # food_probs += list(probs)
                    
                    probs = np.array(probs)
                    non_food_idx = probs[:,0] < threshold
                    img_ids = np.array(img_ids)
                    # non_food_ids += list(img_ids[non_food_idx])
                    # probs_80s = np.logical_and(probs[:,0] < 0.85, probs[:,0] > 0.8)
                    # img_80s = img_ids[probs_80s]
                    for id in img_ids[non_food_idx]:
                        path = os.path.join(split, id[0], id[1], id[2], id[3], id)
                        paths.append(path)
        
        total_data = len(data_loaders['train']) + len(data_loaders['test']) + len(data_loaders['test'])
        print("Total number of non_food: ", len(paths))
        print("Total number of datapoints: ", total_data*args.batch_size)
        # food_probs = np.array(food_probs)
        # print("Mean of food probs: ", np.mean(food_probs))
        # print("Std of food_probs: ", np.std(food_probs))

        with open('/home/donghee/inversecooking/non_food.json', 'w') as f:
            json.dump(paths, f, indent=4)
             

    else:

        ingrs_vocab = pickle.load(open('/home/donghee/inversecooking/data/quantity_recipe1m_vocab_ingrs.pkl', 'rb'))
        idx2ingr = ingrs_vocab.idx2word
        ious = []   
        n_test = 0  
        test_losses = []
        model.eval()

        for batch in data_loaders['test']:

            if batch is None:
                continue
                        
            n_test += batch[0].shape[0]
            img_inputs, ingr_gt, quantity_gt, recipe_ids, img_ids = batch

            losses, pred_quantity = model.forward(img_inputs, ingr_gt, quantity_gt, val = True)
            loss = losses['quantity_loss']
            # loss = 1000.0*losses['ingr_loss'] + 1000.0*losses['quantity_loss'] + 1.0*losses['eos_loss'] + 1.0*losses['card_penalty']

            test_losses.append(loss.item())
            if np.random.rand() < 0.2:
                check_example(logging, idx2ingr, ingr_gt[0], quantity_gt[0], pred_quantity[0], recipe_ids[0])
        
        print("n_test: ", n_test)
        print("test loss: ", sum(test_losses)/len(test_losses))


def check_example(logging, idx2ingr, ingr_gt, quantity_gt, pred_quantity, img_id, pred_ingr):

    i = 0
    ingr_true = []
    quantity_gt_short = []
    quantity_pred_short = []
    while ingr_gt[i] != 1487 and ingr_gt[i] != 0:
        idx = ingr_gt[i].item() ## 30
        ingr_true.append(idx2ingr[idx][0]) ## elbow macaroni
        if quantity_gt is not None and pred_quantity is not None:
            quantity_gt_short.append(quantity_gt[idx].item()) ## 1.5
            quantity_pred_short.append(pred_quantity[idx].item()) ## 1.76
        i += 1
    
    ingr_pred = []
    i = 0
    while pred_ingr[i] != 1487 and pred_ingr[i] != 0:
        idx = pred_ingr[i].item() ## 30
        ingr_pred.append(idx2ingr[idx][0]) ## elbow macaroni
        i += 1

    logging.info("###################")
    logging.info(f"Image ID: {img_id}")
    logging.info(f"True ingredient: {ingr_true}")
    logging.info(f"Predicted ingredient: {ingr_pred}")
    if quantity_gt is not None and pred_quantity is not None:
        logging.info(f"True quantity: {quantity_gt_short}")
        logging.info(f"Predicted quantity: {quantity_pred_short}")
        logging.info(f"Predicted quantity (entire): {pred_quantity.item()}")
    logging.info("###################")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe1m_dir', type=str, default='/home/donghee/inversecooking/recipe1M',
                        help='path to the recipe1m dataset')
    parser.add_argument('--save_dir', type=str, default='/home/donghee/inversecooking/results',
                        help='path where the checkpoints will be saved')
    parser.add_argument('--num_epochs', type=int, default=400,
                        help='maximum number of epochs')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='frequency to validate') ## 1로 하자!!
    # parser.add_argument('--mode', type=str, default='clip_test',
    #                     help='train or test')
    

    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly or center cropping images')

    parser.add_argument('--image_size', type=int, default=256, help='size to rescale images')

    parser.add_argument('--aux_data_dir', type=str, default='/home/donghee/inversecooking/data',
                        help='path to other necessary data files (eg. vocabularies)')
    
    parser.add_argument('--max_eval', type=int, default=4096,
                        help='number of validation samples to evaluate during training')
    
    parser.add_argument('--batch_size', type=int, default=150) ##
    parser.add_argument('--maxseqlen', type=int, default=15,
                        help='maximum length of each instruction')

    parser.add_argument('--maxnuminstrs', type=int, default=10,
                        help='maximum number of instructions')

    parser.add_argument('--maxnumims', type=int, default=5,
                        help='maximum number of images per sample')

    parser.add_argument('--maxnumlabels', type=int, default=20,
                        help='maximum number of ingredients per sample')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default= 1e-03,
                    help='base learning rate') ## learning rate 0.0001 에서 바꿈
    

    parser.add_argument('--project_name', type=str, default='',
                        help='name of the directory where models will be saved within save_dir')

    parser.add_argument('--model_name', type=str, default='',
                        help='save_dir/project_name/model_name will be the path where logs and checkpoints are stored')

    parser.add_argument('--transfer_from', type=str, default='',
                        help='specify model name to transfer from')

    parser.add_argument('--suff', type=str, default='',
                        help='the id of the dictionary to load for training')

    parser.add_argument('--image_model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101',
                                                                                 'resnet152', 'inception_v3'])


    parser.add_argument('--log_step', type=int , default=10, help='step size for printing log info')


    parser.add_argument('--scale_learning_rate_cnn', type=float, default=1.0,
                        help='lr multiplier for cnn weights') ## default=0.01이었는데 ingredient from scratch는 1.0

    parser.add_argument('--lr_decay_rate', type=float, default=0.99,
                        help='learning rate decay factor')

    parser.add_argument('--lr_decay_every', type=int, default=1,
                        help='frequency of learning rate decay (default is every epoch)')

    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--embed_size', type=int, default=512,
                        help='hidden size for all projections')

    parser.add_argument('--n_att', type=int, default=8,
                        help='number of attention heads in the instruction decoder')

    parser.add_argument('--n_att_ingrs', type=int, default=4,
                        help='number of attention heads in the ingredient decoder')

    parser.add_argument('--transf_layers', type=int, default=16,
                        help='number of transformer layers in the instruction decoder')

    parser.add_argument('--transf_layers_ingrs', type=int, default=4,
                        help='number of transformer layers in the ingredient decoder')


    parser.add_argument('--dropout_encoder', type=float, default=0.3,
                        help='dropout ratio for the image and ingredient encoders')

    parser.add_argument('--dropout_decoder_r', type=float, default=0.3,
                        help='dropout ratio in the instruction decoder')

    parser.add_argument('--dropout_decoder_i', type=float, default=0.3,
                        help='dropout ratio in the ingredient decoder')

    parser.add_argument('--finetune_after', type=int, default=-1,
                        help='epoch to start training cnn. -1 is never, 0 is from the beginning')

    parser.add_argument('--loss_weight', nargs='+', type=float, default=[1.0, 0.0, 0.0, 0.0],
                        help='training loss weights. 1) instruction, 2) ingredient, 3) eos 4) cardinality')


    parser.add_argument('--label_smoothing_ingr', type=float, default=0.1,
                        help='label smoothing for bce loss for ingredients')

    parser.add_argument('--patience', type=int, default=50,
                        help='maximum number of epochs to allow before early stopping')

    parser.add_argument('--es_metric', type=str, default='loss', choices=['loss', 'iou_sample'],
                        help='early stopping metric to track')

    parser.add_argument('--eval_split', type=str, default='val')

    parser.add_argument('--numgens', type=int, default=3)

    parser.add_argument('--greedy', dest='greedy', action='store_true',
                        help='enables greedy sampling (inference only)')
    parser.set_defaults(greedy=False)

    parser.add_argument('--temperature', type=float, default=1.0,
                        help='sampling temperature (when greedy is False)')

    parser.add_argument('--beam', type=int, default=-1,
                        help='beam size. -1 means no beam search (either greedy or sampling)')

    parser.add_argument('--ingrs_only', dest='ingrs_only', action='store_true',
                        help='train or evaluate the model only for ingredient prediction')
    parser.set_defaults(ingrs_only=True)

    parser.add_argument('--recipe_only', dest='recipe_only', action='store_true',
                        help='train or evaluate the model only for instruction generation')
    parser.set_defaults(recipe_only=False)

    parser.add_argument('--log_term', dest='log_term', action='store_true',
                        help='if used, shows training log in stdout instead of saving it to a file.')
    parser.set_defaults(log_term=False)

    parser.add_argument('--notensorboard', dest='tensorboard', action='store_false',
                        help='if used, tensorboard logs will not be saved')
    parser.set_defaults(tensorboard=True)

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from the checkpoint in model_name')
    parser.set_defaults(resume=False)

    parser.add_argument('--nodecay_lr', dest='decay_lr', action='store_false',
                        help='disables learning rate decay')
    parser.set_defaults(decay_lr=True)

    parser.add_argument('--load_jpeg', dest='use_lmdb', action='store_false',
                        help='if used, images are loaded from jpg files instead of lmdb')
    parser.set_defaults(use_lmdb=True)

    parser.add_argument('--get_perplexity', dest='get_perplexity', action='store_true',
                        help='used to get perplexity in evaluation')
    parser.set_defaults(get_perplexity=False)

    parser.add_argument('--use_true_ingrs', dest='use_true_ingrs', action='store_true',
                        help='if used, true ingredients will be used as input to obtain the recipe in evaluation')
    parser.set_defaults(use_true_ingrs=False)

    args = parser.parse_args()

    main(args)
    