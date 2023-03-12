# import torch.utils.data as data
from PIL import Image
import os
# import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
# import torch.utils.data
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# import torch.backends.cudnn as cudnn
# from data_loader import ImagerLoader 
# from args import get_parser
# from trijoint import im2recipe
from tqdm import tqdm
# from torchvision.datasets import CIFAR100
import torch.utils.data as data
# import skimage
# import IPython.display
import matplotlib.pyplot as plt
import lmdb
import pickle
import json
import numpy
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

SEED = 1002

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im

    except:
        # print("...", file=sys.stderr)
        print("==========load error==========path: {}=============".format(path))
        return Image.new('RGB', (224, 224), 'white')

def top_ingr(n = 3000):
    with open('/home/donghee/im2recipe-Pytorch/mydata_title/top_ingr.json') as f:
        top_ingr = json.load(f)

    top_ingr_names = [k for k,v in top_ingr.items() if v >= 4] ### 3부터는 이상한 단어들이 뽑혀서 4이상만 뽑았음.. ## 3815개
    top_ingr_names = top_ingr_names[:n] ## 뒷부분이 좀 이상해서 일단 3000개만 뽑음!

    ##
    top_ingr_names = ["flour", "sugar", "baking powder", "baking soda", "vanilla extract", "cinnamon", "nutmeg", "eggs", "milk", "cream", "cream cheese", "butter", "margarine", "vegetable oil", "olive oil", "honey", "brown sugar", "powdered sugar", "cornstarch", "milk", "cream", "butter", "cheese", "sour cream", "ground beef", "bacon", "chicken breast", "onions", "garlic", "tomatoes", "carrots", "celery", "potatoes", "mushrooms", "zucchini", "parsley", "cilantro", "scallions", "salt", "pepper", "cumin", "chili powder", "cayenne pepper", "paprika", "nutmeg", "oregano", "ginger", "worcestershire sauce", "soy sauce", "ketchup", "dijon mustard", "mayonnaise", "lemon", "orange", "raisins", "water", "white wine", "vinegar", "lemon juice", "orange juice", "chicken broth", "tomato sauce", "tomato paste", "egg yolks", "egg whites", "walnuts", "pecans"]
    top_ingr_names = set(top_ingr_names)
    ##

    top_ingr_len = len(top_ingr_names)
    top_ingr_dict = {} ## 'ingr' : num
    for i, ingr in enumerate(top_ingr_names):
        top_ingr_dict[ingr] = i
    
    num_ingr_dict = {} ## 'num' : ingr
    for ingr, num in top_ingr_dict.items():
        num_ingr_dict[num] = ingr

    return top_ingr_dict, num_ingr_dict

class recipe_dataset(data.Dataset):
    def __init__(self, img_path, text_path, preprocess, partition, ingr_num_dict, loader = default_loader): ##
        # partition = 'val'
        self.env = lmdb.open(os.path.join(text_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        self.partition = partition
        with open(os.path.join(text_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)
    
        self.img_path = img_path
        self.loader = loader 
        self.preprocess = preprocess
        self.ingr_num_dict = ingr_num_dict

   
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):

        # if self.partition == 'train':
        #     match = np.random.uniform() > self.mismtch
        # elif self.partition == 'val' or self.partition == 'test':
        #     match = True
        # else:
        #     raise 'Partition name not well defined'

        # target = match and 1 or -1

        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode('latin1')) 

        ## image    
        sample = pickle.loads(serialized_sample,encoding='latin1')
        imgs = sample['imgs']

        if self.partition == 'train':
                # We do only use the first five images per recipe during training
                ## len(imgs) 체크.. 과연 ...
            imgIdx = np.random.choice(range(min(5, len(imgs))))
        else:
            imgIdx = 0

        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        # path = os.path.join(self.imgPath, loader_path, imgs[imgIdx]['id'])

        try:
            path = os.path.join(self.img_path, self.partition, loader_path, imgs[imgIdx]['id'])
        except:
            print("## path error## loader_path: {}, imgIdx: {}".format(loader_path, imgIdx))


        ## ingredients
        ingrs = sample['ingrs']

        threshold = 50

        entry_key_ingr = set()

        for ingr in ingrs: ## TODO scorer 바꿔야 할듯..
            match, score = process.extractOne(ingr, self.ingr_num_dict.keys(), scorer = fuzz.token_sort_ratio)
            if score >= threshold:
                entry_key_ingr.add(match)
        
        key_ingr = [self.ingr_num_dict[ingr] for ingr in entry_key_ingr]
        key_tensor = torch.tensor(key_ingr, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(key_tensor, len(self.ingr_num_dict))
        labels = torch.sum(one_hot, dim=0) #### keepdims = True    



        #### key ingre ## TODO 여기부터 수정!
        # top_ingr_names = set(self.ingr_num_dict.keys()) ## top ingr set
        # # entry_ingr = set(ingrs.split(', ')) ## entry ingr list -> set
        # entry_ingr = set(ingrs)
        # entry_key_ingr = top_ingr_names.intersection(entry_ingr) ## set. intersection
        # key_ingr = [self.ingr_num_dict[ingr] for ingr in entry_key_ingr] ## ingr -> int ## Ground truth?

        # key_tensor = torch.tensor(key_ingr, dtype=torch.long) ##
        # one_hot = torch.nn.functional.one_hot(key_tensor, len(self.ingr_num_dict))
        # labels = torch.sum(one_hot, dim=0) #### keepdims = True        
        ####

        img = self.loader(path)
        img = self.preprocess(img)
        
        ## title
        titles = sample['title']

        ## recipe id
        rec_id = self.ids[index] ## = img_id

        ## image id
        img_id = self.ids[index]
        # if target == -1:
        #     img_id = self.ids[rndindex]
        # else:
        #     img_id = self.ids[index]

        return img, labels, titles, rec_id, img_id


def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        # p.grad.data = p.grad.data.float() 



def main():
    ###########################
    img_path = '/home/donghee/im2recipe-Pytorch/data/images/'
    data_path = '/home/donghee/im2recipe-Pytorch/mydata_ingr/'
    batch_size = 320 # 160
    workers = 8
    EPOCH = 30
    val_freq = 5
    best_val = float('inf')
    # multiple_gpu = False
    mode = 'train' ## 'train' or 'test' ## test - no finetuning.
    # resume = False

    n = 100 ## num of top ingredients (num of classes)


    project_name = 'only_quantity'
    model_name = ''
    quantity_only = True
    ###########################

    

    clip_model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

    # torch.save(model.state_dict(), 'clip_off_the_shelve.pt')
    # model = clip.model.build_model(torch.load('clip_off_the_shelve.pt'))
    
    # model.context_length = context_length
    # model['model_state_dict']['context_length'] = 100 ## not working

    # torch.distributed.init_process_group(backend='nccl')
    
    checkpoint = torch.load("/home/donghee/CLIP/snapshots6/model_epoch001_tloss-1.227(val_medR2.000).pth.tar")
    clip_model.load_state_dict(checkpoint['model_state_dict'])
    convert_models_to_fp32(clip_model)

    clip_model = clip_model.to(device)
    # clip_model = torch.nn.parallel.DistributedDataParallel(clip_model)
        # epoch_resume = checkpoint['epoch']
    
    # else:
    #     epoch_resume = 0
    # class Classifier(nn.Module):
    #     def __init__(self):
    #         super(Classifier, self).__init__()
    #         self.fc1 = nn.Linear(512, 512)
    #         self.fc2 = nn.Linear(512, 4000)
    #         self.sigmoid = nn.Sigmoid()

    #     def forward(self, x):
    #         x = self.fc1(x)
    #         x = self.fc2(x)
    #         x = self.sigmoid(x)
    #         return x

    ingr_num_dict, num_ingr_dict = top_ingr(n)

    class Classifier(nn.Module):
        def __init__(self, num_classes):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(512, 200) ### TODO 항상 바꿔줘!!
            self.fc2 = nn.Linear(200, num_classes)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            # x = self.sigmoid(x)
            return x

    classifier = Classifier(len(ingr_num_dict))
    # classifier = torch.nn.Dataparallel(classifier.to(all_device), device_ids=[0,1,2,3])
    # classifier = torch.nn.Dataparallel(classifier)
    classifier = classifier.to(device)
    # classifier = torch.nn.parallel.DistributedDataParallel(classifier)

    for param in clip_model.parameters():
        param.requires_grad = False
    
    # criterion = nn.BCEWithLogitsLoss ## 이건 sigmoid를 거치지 않은 값을 그대로 전달
    criterion = nn.MultiLabelSoftMarginLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 1e-03) ### lr??
    # optimizer = optimizer.to(device)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=True) 

    

    
    # model_text = TextCLIP(model)
    # model_image = ImageCLIP(model)
    # model_text = torch.nn.parallel.DistributedDataParallel(model_text)
    # model_image = torch.nn.parallel.DistributedDataParallel(model_image)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225]) ## 일반적인 imagenet 정규화


    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]) ## From CLIP preprocess

    transform_train = transforms.Compose([
            transforms.Resize(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(256), # we get only the center of that rescaled
            transforms.RandomCrop(224), # random crop within the center crop 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_val = transforms.Compose([\
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    ## Dataloader 에서 num_worker란, parallelize the loading of the dataset
    ## pin_memory=True: enables the use of pinned memory, which can speed up data transfer between CPU and GPU memory. -- multiple GPU 사용할 때는 True로 해놓기
    recipe_train = recipe_dataset(img_path = img_path, text_path = data_path, preprocess= transform_train, partition= 'train', ingr_num_dict=ingr_num_dict)
    # recipe_train = torch.utils.data.Subset(recipe_train, range(10000)) ### subset
    train_loader = torch.utils.data.DataLoader(recipe_train, batch_size = batch_size, shuffle=True, num_workers=workers, pin_memory=True) 
    print('Training loader prepared.')

    recipe_val = recipe_dataset(img_path = img_path, text_path = data_path, preprocess= transform_val, partition= 'val', ingr_num_dict=ingr_num_dict)
    # recipe_val = torch.utils.data.Subset(recipe_val, range(10000)) ### subset
    val_loader = torch.utils.data.DataLoader(recipe_val, batch_size = batch_size, shuffle=False, num_workers=workers, pin_memory=True) 
    print('Validation loader prepared.')

    
    ### 여기서 weight_decay, lr 바꿔보기
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6) ## lr = 5e-5

    # 'optimizer_state_dict': optimizer.state_dict() ## optimizer 파라미터가 이폭마다 달라지면 아마 optimizer state도 따로 로드해줘야 할듯. resume 시에!!

    # add your own code to track the training progress.

    # losses = [] # train loss. per epoch

    if mode == 'train':
        for epoch in tqdm(range(EPOCH)):
            print("Epoch: ", epoch)
            # print("lr: ", optimizer.param_groups[0]['lr'])

            for batch in tqdm(train_loader) :
                images, labels, titles, rec_id, img_id = batch 
                images = images.to(device)
                labels = labels.to(device)

                img_embedding = clip_model.encode_image(images)
                # img_embedding = img_embedding.to(torch.float) ## fp32로 바꾸기는 했는데, clip_model의 파라미터들을 다 fp32로 바꾸는 게 나을듯..

                output = classifier(img_embedding)

                loss = criterion(output, labels) ## not sure.. output이 logits 여야 하는지, sigmoid 적용한 probability이어야 하는지..
                # losses.append(loss.item())

                predicted = (torch.sigmoid(output) > 0.5).float()

                accuracy = jaccard_score(labels.cpu(), predicted.cpu(), average='samples', zero_division = 1.0) ## zero division...
                ## TODO train loss는 내려가는데 accuracy가 안 내려가..ㅎㅎ 다른 accuracy 방식 찾아보기

                ##


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f'==========Epoch {epoch+1}, Loss: {loss.item()}==============')
            scheduler.step()

            if (epoch+1) % val_freq == 0 or epoch == 0:
                print("## VAL ##")

                with torch.no_grad():
                    val_loss = 0.0
                    for batch in tqdm(val_loader):
                        images, labels, titles, rec_id, img_id= batch 
                        images = images.to(device)
                        labels = labels.to(device)

                        img_embedding = clip_model.encode_image(images)
                        img_embedding = img_embedding.to(torch.float) 

                        output = classifier(img_embedding)

                        loss = criterion(output, labels)
                        val_loss += loss

                        predicted = (torch.sigmoid(output) > 0.5).float()
                        val_accuracy = jaccard_score(labels.cpu(), predicted.cpu(), average='samples', zero_division=1.0) 


                        if np.random.uniform() > 0.7: ## val에서 30%만 예시로 뽑아보기
                            print("Title: ", titles[2])
                            indices = torch.where(labels[2] == 1)[0].tolist()
                            true_ingr = [num_ingr_dict[index] for index in indices] ## TODO recipedataset에서 labels만 return 할 게 아니라 ingrs 자체도 리턴. 그래야 제대로 보지..
                            indices = torch.where(predicted[2] == 1)[0].tolist()
                            predicted_ingr = [num_ingr_dict[index] for index in indices]
                            print("True ingr: ", true_ingr)
                            print("Predicted ingr: ", predicted_ingr)

                    
                val_loss = val_loss / len(val_loader)
                is_best = val_loss < best_val
                best_val = min(val_loss, best_val)

                print("** VAL accuracy: ", val_accuracy)
                print('** Validation: %f (current)' %(val_loss))
                print('** Validation: %f (best)' %(best_val))

                # if is_best:
                filename = '/home/donghee/CLIP/snapshots0_classifier/' + 'model_epoch%03d_tloss-%.3f(val_acc%.3f).pth.tar' % (epoch, best_val, val_accuracy)
                torch.save({
                'epoch': epoch +1,
                'clip_model_state_dict': clip_model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'classifier_optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val,
                'curr_val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'lr': optimizer.param_groups[0]['lr']
                }, filename)
                print("Save model")


                    ## TODO is_best 일 떄 val에서 label 이랑 predicted 다 stack 해서 confusion matrix 뽑기..
                    # conf_matrix = multilabel_confusion_matrix(labels.cpu(), predicted.cpu())
                    
                    # for i, m in enumerate(conf_matrix):
                    #     sns.heatmap(m, annot=True, cmap='Blues', fmt='g')
                    #     plt.xlabel('Predicted Label')
                    #     plt.ylabel('True Label')
                    #     recall = m[1][1] / (m[1][0]+m[1][1])
                    #     plt.savefig('/home/donghee/CLIP/confusion_matrix/cm_{}_recall_{:.3f}'.format(num_ingr_dict[i], recall))
                    
                    # print("save confusion matrix")


    elif mode =='test':

        print("TEST mode start")
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, labels, titles, rec_id, img_id = batch 
                images = images.to(device)
                labels = labels.to(device)

                img_embedding = clip_model.encode_image(images)
                img_embedding = img_embedding.to(torch.float) 

                output = classifier(img_embedding)

                loss = criterion(output, labels)
                val_loss += loss

                predicted = (torch.sigmoid(output) > 0.5).float()
                val_accuracy = jaccard_score(labels.cpu(), predicted.cpu(), average='samples', zero_division=1.0) 

                conf_matrix = multilabel_confusion_matrix(labels.cpu(), predicted.cpu())

                for i, m in enumerate(conf_matrix):
                    sns.heatmap(m, annot=True, cmap='Blues', fmt='g')
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    recall = m[1][1] / (m[1][0]+m[1][1])
                    plt.savefig('/home/donghee/CLIP/confusion_matrix/cm_{}_recall_{:.3f}'.format(num_ingr_dict[i], recall))


if __name__ == '__main__':
    main()
