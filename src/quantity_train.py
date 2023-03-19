from args import get_parser
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import random
import pickle
from quantity_data_loader import get_loader
from build_vocab import Vocabulary
from model import get_model
from torchvision import transforms
import sys
import json
import time
import torch.backends.cudnn as cudnn
from utils.tb_visualizer import Visualizer
from model import mask_from_eos, label2onehot
from utils.metrics import softIoU, compute_metrics, update_error_types
import random
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

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

def take_valid_quantity(ingr_ids, quantity):
    pred_quantity = np.zeros_like(quantity)
    i = 0 
    while ingr_ids[i] != 0 and ingr_ids[i] != 1478: ## TODO hardcode 바꾸기
        pred_quantity[int(ingr_ids[i])] = quantity[int(ingr_ids[i])]
        i += 1
    
    return pred_quantity

def print_check(img_id, ingr_gt, ingr_ids_greedy, quantity_gt, pred_quantity):
    print("===================")
    print("- img id: ", img_id)
    print("- ingredient GT: ", ingr_gt)
    print("- ingredient prediction: ", ingr_ids_greedy)
    print("- quantity GT: ", quantity_gt)
    print("- quantity prediction: ", pred_quantity)
    print("===================")

class Quantity_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x) ## Q) regression 에서는 마지막에 nonlinearliyt layer 없나?
        return x


def main(args):

    ####################
    project_name = 'quantity_extended'
    model_name = ''
    quantity_only = True
    ####################

    # Create model directory & other aux folders for logging
    where_to_save = os.path.join(args.save_dir, project_name, model_name) ## save_dir = ../results
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    logs_dir = os.path.join(where_to_save, 'logs')
    tb_logs = os.path.join(args.save_dir, project_name, 'tb_logs', model_name)
    make_dir(where_to_save)
    make_dir(logs_dir)
    make_dir(checkpoints_dir)
    make_dir(tb_logs)
    if args.tensorboard:
        logger = Visualizer(tb_logs, name='visual_results')

    # check if we want to resume from last checkpoint of current model
    if args.resume:
        args = pickle.load(open(os.path.join(checkpoints_dir, 'args.pkl'), 'rb'))
        args.resume = True
    
    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(logs_dir, 'train.log'))
        sys.stdout = open(os.path.join(logs_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'train.err'), 'w')
    
    # print(args)
    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))

    # patience init
    curr_pat = 0

    # Build data loader
    data_loaders = {}
    datasets = {}

    data_dir = args.recipe1m_dir

    for split in ['train', 'val']:

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
        data_loaders[split], datasets[split] = get_loader(data_dir, args.aux_data_dir, split,
                                                          args.maxseqlen,
                                                          args.maxnuminstrs,
                                                          args.maxnumlabels,
                                                          args.maxnumims,
                                                          transform, args.batch_size,
                                                          shuffle=split == 'train', num_workers=args.num_workers,
                                                          drop_last=True,
                                                          max_num_samples=max_num_samples,
                                                          use_lmdb=args.use_lmdb,
                                                          suff=args.suff)


    ingr_vocab_size = datasets[split].get_ingrs_vocab_size() ## 1488
    # instrs_vocab_size = datasets[split].get_instrs_vocab_size()

    # Build the model
    quantity_model = Quantity_MLP(input_size = 512*49, hidden_size = 1024, output_size = ingr_vocab_size-1)
    inverse_model = get_model(args, ingr_vocab_size, 23231) ## TODO hardcode here
    keep_cnn_gradients = False
    inverse_model_path = os.path.join('/home/donghee/inversecooking/data', 'modelbest.ckpt')
    inverse_model.load_state_dict(torch.load(inverse_model_path, map_location=map_loc))

    for p in inverse_model.parameters():
        p.requires_grad = False
    
    for p in inverse_model.image_encoder.linear.parameters():
        p.requires_grad = True
    
    for p in inverse_model.ingredient_decoder.parameters():
        p.requires_grad = True

    decay_factor = 1.0

    quantity_model = quantity_model.to(device)
    inverse_model = inverse_model.to(device)

    params = list(inverse_model.ingredient_decoder.parameters())
    params_quantity = list(quantity_model.parameters())
    params_linear = list(inverse_model.image_encoder.linear.parameters())

    params += params_quantity
    params += params_linear

    # only train the linear layer in the encoder if we are not transfering from another model
    # if args.transfer_from == '':
    #     params += list(model.image_encoder.linear.parameters())
    # params_cnn = list(model.image_encoder.resnet.parameters())

    # print ("CNN params:", sum(p.numel() for p in params_cnn if p.requires_grad))
    print ("ingredient decoder params:", sum(p.numel() for p in params if p.requires_grad))
    print ("quantity decoder params:", sum(p.numel() for p in params_quantity if p.requires_grad))
    print ("image linear params:", sum(p.numel() for p in params_linear if p.requires_grad))

    # start optimizing cnn from the beginning
    # if params_cnn is not None and args.finetune_after == 0:
    #     optimizer = torch.optim.Adam([{'params': params}, {'params': params_cnn,
    #                                                        'lr': args.learning_rate*args.scale_learning_rate_cnn}],
    #                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    #     keep_cnn_gradients = True
    #     print ("Fine tuning resnet")
    # else:
    #     optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # if args.resume:
    #     model_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'model.ckpt')
    #     optim_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'optim.ckpt')
    #     optimizer.load_state_dict(torch.load(optim_path, map_location=map_loc))
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(device)
    #     model.load_state_dict(torch.load(model_path, map_location=map_loc))
    
    ##loads CNN encoder from transfer_from model(생략)

    # if device != 'cpu' and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # model = model.to(device)
    cudnn.benchmark = True

    if not hasattr(args, 'current_epoch'):
        args.current_epoch = 0

    es_best = 10000 if args.es_metric == 'loss' else 0
    # Train the model
    start = args.current_epoch
    for epoch in tqdm(range(start, args.num_epochs)):

        # save current epoch for resuming
        if args.tensorboard:
            logger.reset()

        args.current_epoch = epoch
        # increase / decrase values for moving params
        if args.decay_lr:
            frac = epoch // args.lr_decay_every
            decay_factor = args.lr_decay_rate ** frac
            new_lr = args.learning_rate*decay_factor
            print ('Epoch %d. lr: %.5f'%(epoch, new_lr))
            set_lr(optimizer, decay_factor)

        # if args.finetune_after != -1 and args.finetune_after < epoch \
        #         and not keep_cnn_gradients and params_cnn is not None:

        #     print("Starting to fine tune CNN")
        #     # start with learning rates as they were (if decayed during training)
        #     optimizer = torch.optim.Adam([{'params': params},
        #                                   {'params': params_cnn,
        #                                    'lr': decay_factor*args.learning_rate*args.scale_learning_rate_cnn}],
        #                                  lr=decay_factor*args.learning_rate)
        #     keep_cnn_gradients = True
        
        for split in ['train', 'val']:
            print("======== Epoch: ", epoch, ", split: ", split)

            if split == 'train':
                inverse_model.train()
                quantity_model.train()
            else:
                inverse_model.eval()
                quantity_model.eval()
            total_step = len(data_loaders[split])
            loader = iter(data_loaders[split])

            total_loss_dict = {'ingr_loss': [],
                               'eos_loss': [], 'loss': [],
                               'iou': [], 'iou_sample': [],
                               'f1': [],
                               'card_penalty': [],
                               'quantity_loss': []}

            error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0,
                           'tp_all': 0, 'fp_all': 0, 'fn_all': 0}

            torch.cuda.synchronize()
            start = time.time()

            for i in range(total_step):
                ##args. quantity only 추가^^
                img_inputs, ingr_gt, quantity_gt, img_ids, paths= loader.next()

                ingr_gt = ingr_gt.to(device)
                quantity_gt = quantity_gt.to(device)
                img_inputs = img_inputs.to(device)
                
                loss_dict = {}

                if split == 'val':
                    with torch.no_grad():
                        losses = inverse_model(img_inputs, ingr_gt, quantity_gt)
                        
                        outputs = inverse_model(img_inputs, ingr_gt, quantity_gt, sample=True) ## TODO - sample 수정 (quantity 나올 수 있게)

                        ingr_ids_greedy = outputs['ingr_ids']

                        mask = mask_from_eos(ingr_ids_greedy, eos_value=0, mult_before=False)
                        ingr_ids_greedy[mask == 0] = ingr_vocab_size-1
                        pred_one_hot = label2onehot(ingr_ids_greedy, ingr_vocab_size-1)
                        target_one_hot = label2onehot(ingr_gt, ingr_vocab_size-1)
                        iou_sample = softIoU(pred_one_hot, target_one_hot)
                        iou_sample = iou_sample.sum() / (torch.nonzero(iou_sample.data).size(0) + 1e-6)
                        loss_dict['iou_sample'] = iou_sample.item()

                        # pred_quantity = take_valid_quantity(ingr_ids_greedy, outputs['pred_quantity'])

                        # if random.random() < 0.1:
                        #     print_check(img_ids[0], ingr_gt[0], ingr_ids_greedy[0], quantity_gt[0], pred_quantity[0])

                        update_error_types(error_types, pred_one_hot, target_one_hot)

                        del outputs, pred_one_hot, target_one_hot, iou_sample

                else:
                    losses = inverse_model(img_inputs, ingr_gt, quantity_gt,
                                   keep_cnn_gradients=keep_cnn_gradients)
                    
                # recipe_loss = 0
                ingr_loss = losses['ingr_loss']
                ingr_loss = ingr_loss.mean()
                loss_dict['ingr_loss'] = ingr_loss.item()

                eos_loss = losses['eos_loss']
                eos_loss = eos_loss.mean()
                loss_dict['eos_loss'] = eos_loss.item()

                iou_seq = losses['iou']
                iou_seq = iou_seq.mean()
                loss_dict['iou'] = iou_seq.item()

                card_penalty = losses['card_penalty'].mean()
                loss_dict['card_penalty'] = card_penalty.item()

                quantity_loss = losses['quantity_loss']
                quantity_loss = quantity_loss.mean()
                loss_dict['quantity_loss'] = quantity_loss.item()

                # loss_weight = [1000.0, 1000.0, 1.0, 1.0]
                loss = args.loss_weight[0] * ingr_loss + args.loss_weight[1] * quantity_loss \
                       + args.loss_weight[2]*eos_loss + args.loss_weight[3]*card_penalty
                
                # loss_weight = torch.tensor([1000.0, 1000.0, 1.0, 1.0], dtype = torch.float64)
                # loss = loss_weight[0] * ingr_loss + loss_weight[1] * quantity_loss \
                #        + loss_weight[2]*eos_loss + loss_weight[3]*card_penalty

                loss_dict['loss'] = loss.item()

                for key in loss_dict.keys():
                    total_loss_dict[key].append(loss_dict[key])

                if split == 'train':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Print log info
                if args.log_step != -1 and i % args.log_step == 0:
                    elapsed_time = time.time()-start
                    lossesstr = ""
                    for k in total_loss_dict.keys():
                        if len(total_loss_dict[k]) == 0:
                            continue
                        this_one = "%s: %.4f" % (k, np.mean(total_loss_dict[k][-args.log_step:]))
                        lossesstr += this_one + ', '
                    # this only displays nll loss on captions, the rest of losses will be in tensorboard logs
                    strtoprint = 'Split: %s, Epoch [%d/%d], Step [%d/%d], Losses: %sTime: %.4f' % (split, epoch,
                                                                                                   args.num_epochs, i,
                                                                                                   total_step,
                                                                                                   lossesstr,
                                                                                                   elapsed_time)
                    print(strtoprint)

                    if args.tensorboard:
                        # logger.histo_summary(model=model, step=total_step * epoch + i)
                        logger.scalar_summary(mode=split+'_iter', epoch=total_step*epoch+i,
                                              **{k: np.mean(v[-args.log_step:]) for k, v in total_loss_dict.items() if v})

                    torch.cuda.synchronize()
                    start = time.time()
                del loss, losses, img_inputs

            if split == 'val' and not args.recipe_only:
                ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'f1_ingredients': [], 'dice': []}
                compute_metrics(ret_metrics, error_types,
                                ['accuracy', 'f1', 'jaccard', 'f1_ingredients', 'dice'], eps=1e-10,
                                weights=None)
                total_loss_dict['f1'] = ret_metrics['f1']
            
            if args.tensorboard:
                # 1. Log scalar values (scalar summary)
                logger.scalar_summary(mode=split,
                                      epoch=epoch,
                                      **{k: np.mean(v) for k, v in total_loss_dict.items() if v})

        
        # Save the model's best checkpoint if performance was improved
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

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)








