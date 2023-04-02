import json
import sys
sys.path.append('/home/donghee/CLIP')
import clip
import os
import torch
from PIL import Image
import wget
import subprocess
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import urllib.request
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Recipe1M(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        filename = os.path.basename(path)
        return image, filename

def detect_img(model, preprocess, dataloader, text, layer2_ids, ids, img_ids, im2id, im2url, threshold, num):

    with open('/home/donghee/inversecooking/id2probs.json', 'r') as f:
        id2probs = json.load(f)
    # dir_path = os.path.join('/home/donghee/inversecooking/recipe1M+', i)

    # id2probs = {}## 'id': {'img_id': img_id, 'probs': probs}
    original_img_ids_len = len(img_ids)
    
    ## for batch in data_loader: batch_size 1024
    ## batch 로 로드, images, img_ids
    ## probs = img_prob(model, preprocess, text, images)
    ## probs > threshold
    ## valid_img_ids = img_ids[probs>threshold]
    ## for img_id in valid_img_ids:
    ## id = im2id[img_id] @@
    cnt = 0
    for i, batch in enumerate(tqdm(dataloader)):
        images, batch_img_ids = batch
        batch_img_ids = np.array(batch_img_ids)
        cnt += len(batch_img_ids)

        probs = img_prob(model, images, text)
        torch.cuda.synchronize()
        idx = probs[:, 0] > threshold
        valid_img_ids = batch_img_ids[idx]
        probs = probs[idx][:,0]

        for img_id, prob in zip(valid_img_ids, probs):
            id = im2id[img_id]
            if id in layer2_ids:
                continue

            if id not in ids and prob > threshold: ## 첫 추가

                id2probs[id] = {
                    'img_id': img_id,
                    'prob': prob
                }

                img_ids.add(img_id)
                ids.add(id)
                
            elif id in ids and prob > id2probs[id]['prob']: ## update, better image
                img_ids.remove(id2probs[id]['img_id'])
                img_ids.add(img_id)
                id2probs[id]['img_id'] = img_id
                id2probs[id]['prob'] = prob          
    
    assert len(id2probs.keys()) == len(set(id2probs.keys())) ## ids 중복 없어야 함
    intersection = layer2_ids & set(id2probs.keys())
    assert len(intersection) == 0
    assert len(ids) == len(img_ids)

    print(f'** num: {num}, # detected images (total): {len(img_ids)-original_img_ids_len} ({cnt}), detected ids (original layer2 ids): {len(ids)}({len(layer2_ids)})')

    with open('/home/donghee/inversecooking/id2probs.json', 'w') as f:
        json.dump(id2probs, f, indent=4)
    
    return ids, img_ids

def store_new_layer(id2probs):
    with open('/home/donghee/inversecooking/recipe1M/layer2.json', 'r') as f:
        layer2 = json.load(f)
    
    for id, images in id2probs.items():
        layer2.append({
            'id': id,
            'images':[{
                'id':images['img_id'],
                'url':im2url[images['img_id']]
            }]
        })

    new_layer2_ids = [entry['id'] for entry in layer2]
    assert len(new_layer2_ids) == len(set(new_layer2_ids))

    with open('/home/donghee/inversecooking/layer2_download.json', 'w') as f:
        json.dump(layer2, f, indent=4)


def img_prob(model, images, text):
    # image = Image.open(img_path).convert("RGB")
    # image = preprocess(image).cuda()
    # image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        # logits_per_image, logits_per_text, image_embedding, text_embedding = model(images, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        images = images.to(device)
        text = text.to(device)
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(text).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).detach().cpu().numpy()
        del image_features, text_features
    
    return np.array(probs)
    

def delete_img(img_ids, delete_paths):
    ## for os walk, if not in img_ids, delete
    ## print how many images left
    # dir_path = os.path.join('/home/donghee/inversecooking/recipe1M+', i)
    remove_cnt = 0
    for path in delete_paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file in img_ids:
                    continue
                file_path = os.path.join(root, file)
                os.remove(file_path)
                remove_cnt += 1
    
    print("* removed # images: ", remove_cnt)


def new_layer_setup(layer2):
    new_layer = []
    img_ids = set()
    layer2_ids = set()
    for entry in layer2:
        if 'images' in entry and len(entry['images']) != 0:
            layer2_ids.add(entry['id'])
            new_layer.append({
                'id': entry['id'],
                'images': entry['images']
            })
            for image in entry['images']:
                img_ids.add(image['id'])
    
    print("original layer2 # layer2_ids: ", len(layer2_ids))
    print("len of new_layer(remove no images): ", len(new_layer))
    print("len of original layer2: ", len(layer2))
    print("len of original layer2 # images: ", len(img_ids))
    
    with open('/home/donghee/inversecooking/layer2_download.json', 'w') as f: ## 최종 layer2
        json.dump(new_layer, f, indent=4)
    
    json_img_ids = json.dumps(list(img_ids))
    json_layer2_ids = json.dumps(list(layer2_ids))

    with open('/home/donghee/inversecooking/layer2_img_ids.json', 'w') as f:
        f.write(json_img_ids)
    
    with open('/home/donghee/inversecooking/layer2_ids.json', 'w') as f:
        f.write(json_layer2_ids)

    return img_ids, layer2_ids
    
def layer2_setup(layer2p):
    im2id = {} ## img_id to id
    im2url = {} ## img_id to url
    for entry in layer2p:
        if len(entry['images']) != 0:
            for image in entry['images']:
                im2id[image['id']] = entry['id']
                im2url[image['id']] = image['url']
    
    json_im2id = json.dumps(im2id)
    json_im2url = json.dumps(im2url)

    with open('/home/donghee/inversecooking/im2id.json', 'w') as f:
        f.write(json_im2id)
    with open('/home/donghee/inversecooking/im2url.json', 'w') as f:
        f.write(json_im2url)
    
    return im2id, im2url

def count_files(path):
    num_files = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            num_files += 1
    return num_files

def ids_store(ids, img_ids):
    json_img_ids = json.dumps(list(img_ids))
    json_ids = json.dumps(list(ids))

    with open('/home/donghee/inversecooking/ids_download.json', 'w') as f:
        f.write(json_ids)
    with open('/home/donghee/inversecooking/img_ids_download.json', 'w') as f:
        f.write(json_img_ids)

def reporthook(count, block_size, total_size):
    """
    A function to report the download progress.
    """
    progress_size = int(count * block_size)
    percent = int(progress_size * 100 / total_size)
    print(f"\rDownloading... {percent}% [{progress_size}/{total_size}]", end='')

if __name__ == '__main__':

    resume = False
    threshold = 0.90 ####
    batch_size = 256
    num_workers = 16
    num = '0'

    with open('/home/donghee/inversecooking/recipe1M+/layer2+.json', 'r') as f:
        layer2p = json.load(f)
    with open('/home/donghee/inversecooking/recipe1M/layer2.json', 'r') as f:
        layer2 = json.load(f)
    
    if resume:
        with open('/home/donghee/inversecooking/ids_download.json', 'r') as f:
            ids = json.load(f) 
        with open('/home/donghee/inversecooking/img_ids_download.json', 'r') as f:
            img_ids = json.load(f)
        with open('/home/donghee/inversecooking/layer2_ids.json', 'r') as f:
            layer2_ids = json.load(f)
    else:
        # img_ids, layer2_ids = new_layer_setup(layer2)
        with open('/home/donghee/inversecooking/layer2_img_ids.json', 'r') as f:
            img_ids = json.load(f)
        with open('/home/donghee/inversecooking/layer2_ids.json', 'r') as f:
            layer2_ids = json.load(f)
        ids = []
    
    im2id, im2url = layer2_setup(layer2p)
    ids = set(ids)
    img_ids = set(img_ids)

    layer2_ids = set(layer2_ids)
    original_img_ids_len = len(img_ids)

    ## 맨 처음에는 그냥 layer2랑 같은 ids, img_ids로 하고 그 다음 loop 부터 ids_download, img_ids_download에서 불러오면 될듯??
    ## ids/layer2_ids는 따로 둬야 하지만 img_ids 는 하나로 합쳐도 됨!
    
    print("# layer2p+ images : ", len(im2id)) #imgID to ID

    model, preprocess = clip.load("ViT-B/32", device=device)
    # model.cuda().eval()
    text = clip.tokenize(['A photo of a food', 'A photo of non-food']).to(device)

    # if device != 'cpu' and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    ## 여기부터!
    base_path = '/home/donghee/inversecooking/recipe1M+'
    base_url = 'http://data.csail.mit.edu/im2recipe/recipe1M+_images/recipe1M+_'

    nums = ['0', '1', '2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    img_paths = [os.path.join(base_path, num) for num in nums]
    # urls = [base_url+num+'.tar' for num in nums]

    # for i, (num, img_path) in enumerate(zip(nums, img_paths)):
        
    print(f"** Start {num} **")
    url = base_url + num + '.tar'
    tar_file_path = f'/home/donghee/inversecooking/recipe1M+/recipe1M+_{num}.tar'
    # urllib.request.urlretrieve(url, tar_file_path, reporthook)
    subprocess.run(['tar', '-xvf', tar_file_path, '-C', base_path]) ## ## TODO
    os.remove(tar_file_path) ##
    
    img_path = os.path.join(base_path, num)
    dataset = Recipe1M(root=img_path, transform = preprocess)
    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle=False)

    ids, img_ids = detect_img(model, preprocess, dataloader, text, layer2_ids, ids, img_ids, im2id, im2url, threshold, num)
    ids_store(ids, img_ids)

    i = nums.index(num)
    delete_paths = img_paths[:i+1]
    delete_img(img_ids, delete_paths)
    
    remain_files = 0
    for path in delete_paths:
        remain_files += count_files(path)
    
    print(f'** Done {num} **')
    print("# remain images: ", remain_files)

    print(f"*** added # ids with matching image (original layer2): {len(ids)}({len(layer2_ids)}), Total # images extended(original layer2 images): {len(img_ids)}({original_img_ids_len})")
    print("DONE")
    ## TODO store_new_layer
    