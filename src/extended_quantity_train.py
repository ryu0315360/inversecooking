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
import argparse
from utils.output_utils import get_ingrs
import fasldifjoie ##
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

class Inverse_Quantity(nn.Module):
    def __init__(self, inverse_model, quantity_model):
        super().__init__()

        self.inverse_model = inverse_model
        self.quantity_model = quantity_model
        self.quantity_criterion = nn.MSELoss(reduce = False)
        self.quantity_criterion = self.quantity_criterion.to(device)

    def forward(self, img_inputs, ingr_gt, quantity_gt, test=False):

        img_inputs = img_inputs.to(device)
        ingr_gt = ingr_gt.to(device)
        quantity_gt = quantity_gt.to(device)

        losses = self.inverse_model(img_inputs, captions = None, target_ingrs = ingr_gt)
     
        with torch.no_grad():
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

        if test == True:
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
    save_dir = '../results'
    project_name = 'quantity_extended'
    model_name = ''
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

    # quantity_counter(data_loaders)
    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()

    quantity_model = Quantity_MLP(input_size = args.embed_size, hidden_size = 1024, output_size = ingr_vocab_size-1)
    inverse_model = get_model(args, ingr_vocab_size, 23231) ## TODO hardcode here
    keep_cnn_gradients = False
    # inverse_model_path = os.path.join('/home/donghee/inversecooking/data', 'modelbest.ckpt')
    # inverse_model.load_state_dict(torch.load(inverse_model_path, map_location=map_loc))

    for p in inverse_model.parameters():
        p.requires_grad = False

    for p in inverse_model.image_encoder.linear.parameters():
        p.requires_grad = True
    
    # for p in inverse_model.ingredient_decoder.parameters():
    #     p.requires_grad = True
    
    model = Inverse_Quantity(inverse_model, quantity_model)
    decay_factor = 1.0

    # params = list(inverse_model.ingredient_decoder.parameters())
    params = list(model.quantity_model.parameters())
    params_linear = list(model.inverse_model.image_encoder.linear.parameters())
    # params += params_quantity
    params += params_linear

    # print ("ingredient decoder params:", sum(p.numel() for p in params if p.requires_grad))
    print ("quantity decoder params:", sum(p.numel() for p in params if p.requires_grad))
    print ("image linear params:", sum(p.numel() for p in params_linear if p.requires_grad))

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    ## TODO - resume

    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    cudnn.benchmark = True

    if not hasattr(args, 'current_epoch'):
        args.current_epoch = 0

    train_losses = []
    val_losses = []
    best_val = float('inf')
    start = args.current_epoch
    if args.mode == 'train':
        for epoch in tqdm(range(start, args.num_epochs)):
            print("======== EPOCH {} =========".format(epoch))

            if args.tensorboard:
                logger.reset()

            if args.decay_lr:
                frac = epoch // args.lr_decay_every
                decay_factor = args.lr_decay_rate ** frac
                new_lr = args.learning_rate*decay_factor
                print ('Epoch %d. lr: %.5f'%(epoch, new_lr))
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
                img_inputs, ingr_gt, quantity_gt, img_ids, paths = batch

                losses = model.forward(img_inputs, ingr_gt, quantity_gt)

                # loss = losses['quantity_loss']
                ## TODO - ingr / quantity loss scale..
                loss = 1000.0*losses['ingr_loss'] + 1000.0*losses['quantity_loss'] + 1.0*losses['eos_loss'] + 1.0*losses['card_penalty']

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
                        this_one = "%s: %.4f" % (k, np.mean(total_loss_dict[k][-args.log_step:]))
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
                del loss, losses, captions, img_inputs
                    
            
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
            
            torch.cuda.synchronize()
            start = time.time()

            if (epoch+1)%args.val_freq == 0 or epoch == 0:
                model.eval()
                split = 'val'

                with torch.no_grad():
                    val_loss = 0.0
                    for batch in data_loaders['val']:
                        
                        if batch is None:
                            continue
                        
                        n_val += batch[0].shape[0]
                        img_inputs, ingr_gt, quantity_gt, img_ids, paths = batch

                        losses = model.forward(img_inputs, ingr_gt, quantity_gt)
                        loss = losses['quantity_loss']
                        # loss = 1000.0*losses['ingr_loss'] + 1000.0*losses['quantity_loss'] + 1.0*losses['eos_loss'] + 1.0*losses['card_penalty']

                        val_loss += loss.item()
                        
                        if random.random() < 0.1: ## of np.random.uniform()
                            check_example(img_ids[0], ingr_gt[0], quantity_gt[0], pred_quantity[0])
                        for key in losses.keys():
                            total_loss_dict[key].append(losses[key])

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
                        del loss, losses, captions, img_inputs

                if epoch == 0:
                    print("# val data points: ", n_val)
                val_loss = val_loss / len(data_loaders['val'])
                val_losses.append(val_loss)
                print("Val losses: ", val_losses)

                is_best = val_loss < best_val
                best_val = min(val_loss, best_val)
                print("* current val loss: ", val_loss)
                print("* best val_loss: ", best_val)

                if is_best:
                    save_model(model, optimizer, checkpoints_dir, suff='best')
                    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))
                    curr_pat = 0
                    print('Saved checkpoint.')
                    # filename = os.path.join('/home/donghee/inversecooking/results','model3_epoch%03d_tloss-%.3f.pth.tar' % (epoch, best_val))
                    # torch.save({
                    # 'epoch': epoch +1,
                    # 'quantity_model_state_dict': quantity_model.state_dict(),
                    # 'quantity_optimizer_state_dict': optimizer.state_dict(),
                    # 'best_val_loss': best_val,
                    # 'curr_val_loss': val_loss,
                    # 'lr': optimizer.param_groups[0]['lr']
                    # }, filename)
                    # print("Save model")
                else:
                    curr_pat += 1
                
                if curr_pat > args.patience:
                    break

        if args.tensorboard:
            logger.close()   

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
            img_inputs, ingr_gt, quantity_gt, img_ids, paths = batch

            losses, pred_quantity = model.forward(img_inputs, ingr_gt, quantity_gt, test = True)
            loss = losses['quantity_loss']
            # loss = 1000.0*losses['ingr_loss'] + 1000.0*losses['quantity_loss'] + 1.0*losses['eos_loss'] + 1.0*losses['card_penalty']

            test_losses.append(loss.item())
            if np.random.rand() < 0.2:
                check_example(idx2ingr, ingr_gt[0], quantity_gt[0], pred_quantity[0], img_ids[0])
        
        print("n_test: ", n_test)
        print("test loss: ", sum(test_losses)/len(test_losses))


def check_example(idx2ingr, ingr_gt, quantity_gt, pred_quantity, img_id):

    i = 0
    ingr = []
    quantity_gt_short = []
    quantity_pred_short = []
    while ingr_gt[i] != 1487 and ingr_gt[i] != 0:
        idx = ingr_gt[i].item() ## 30
        ingr.append(idx2ingr[idx][0]) ## elbow macaroni
        quantity_gt_short.append(quantity_gt[idx].item()) ## 1.5
        quantity_pred_short.append(pred_quantity[idx].item()) ## 1.76
        i += 1
    
    print("###################")
    print("Image ID: ", img_id)
    print("True ingredient: ", ingr)
    print("True quantity: ", quantity_gt_short)
    print("Predicted quantity: ", quantity_pred_short)
    print("Predicted quantity (entire): ", pred_quantity.item())
    print("###################")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe1m_dir', type=str, default='/home/donghee/inversecooking/recipe1M',
                        help='path to the recipe1m dataset')
    parser.add_argument('--save_dir', type=str, default='/home/donghee/inversecooking/results',
                        help='path where the checkpoints will be saved')
    parser.add_argument('--num_epochs', type=int, default=400,
                        help='maximum number of epochs')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='frequency to validate')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    

    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly or center cropping images')

    parser.add_argument('--image_size', type=int, default=256, help='size to rescale images')

    parser.add_argument('--aux_data_dir', type=str, default='/home/donghee/inversecooking/data',
                        help='path to other necessary data files (eg. vocabularies)')
    
    parser.add_argument('--max_eval', type=int, default=4096,
                        help='number of validation samples to evaluate during training')
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--maxseqlen', type=int, default=15,
                        help='maximum length of each instruction')

    parser.add_argument('--maxnuminstrs', type=int, default=10,
                        help='maximum number of instructions')

    parser.add_argument('--maxnumims', type=int, default=5,
                        help='maximum number of images per sample')

    parser.add_argument('--maxnumlabels', type=int, default=20,
                        help='maximum number of ingredients per sample')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='base learning rate')
    

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


    parser.add_argument('--scale_learning_rate_cnn', type=float, default=0.01,
                        help='lr multiplier for cnn weights')

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
    parser.set_defaults(ingrs_only=False)

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
    