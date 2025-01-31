import pickle
import os
from collections import Counter
import json

def count_ids():
    splits = ['train', 'val', 'test']
    ids = []
    img_cnt = 0
    no_img = 0
    for split in splits:
        dataset = pickle.load(open(os.path.join('/home/donghee/inversecooking/data/recipe1m_'+split+'.pkl'), 'rb'))
        for sample in dataset:
            ids.append(sample['id'])
            img_cnt += len(sample['images'])
            if len(sample['images']) == 0:
                no_img += 1

    print("len of list ids: ", len(ids))
    print("led of set ids: ", len(set(ids)))

    print("img_cnt: ", img_cnt)
    print("no_img: ", no_img)

def quantity_counter(data_loaders):
    ingrs_vocab = pickle.load(open('/home/donghee/inversecooking/data/quantity_recipe1m_vocab_ingrs.pkl', 'rb'))
    idx2ingr = ingrs_vocab.idx2word
    splits = ['train', 'val', 'test']

    ingrs = []
    for split in splits:
        for batch in data_loaders[split]:
            img_inputs, ingr_gt, quantity_gt, img_ids, paths = batch

            for ingr in ingr_gt:
                i=0
                while ingr[i] != 1478 and ingr[i] != 0:
                    ingrs.append(ingr[i].item())
                    i += 1
    
    ingrs_counter = Counter(ingrs)
    ingr_dict = {}
    for k, v in ingrs_counter.items():
        key_ingr = idx2ingr[k][0]
        ingr_dict[key_ingr] = v

    # print(ingr_dict)
    sorted_dict = dict(sorted(ingr_dict.items(), key=lambda item: item[1], reverse=True))
    with open('/home/donghee/inversecooking/jsons/quantity_counter.json', 'w+') as f:
        json.dump(sorted_dict, f, indent=4)

def single_food_counter():
    with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
        dataset_recipe = json.load(f)
    
    with open('/home/donghee/inversecooking/recipe1M/layer2.json', 'r') as f:
        dataset_image = json.load(f)
    
    ids = set()
    for sample in dataset_image:
        ids.add(sample['id'])
    
    del dataset_image
    
    single_wimage = 0
    single = 0
    for sample in dataset_recipe:
        if len(sample['ingredients']) != 1:
            continue
        else: ## single ingredient
            single += 1
            if sample['id'] in ids: ## single and having image
                single_wimage += 1
    
    print("# single ingredient samples: ", single)
    print("# single ingredient with image samples: ", single_wimage)

def count_layer2():
    layer2 = json.load(open(os.path.join('/home/donghee/inversecooking', 'layer2_download.json'), 'r'))
    
    img_cnt = 0
    id_cnt = 0
    id = []
    for entry in layer2:
        id.append(entry['id'])
        if len(entry['images']) != 0:
            id_cnt += 1
        img_cnt += len(entry['images'])
    
    print(img_cnt)
    print(id_cnt)
    print(len(id))
    print(len(set(id)))

def count_nut_layer():
    with open("/home/donghee/inversecooking/recipe1M+/recipes_with_nutritional_info.json", 'r') as f:
        nut_layer = json.load(f)
    print(len(nut_layer))

    ingr2unit = dict()
    for entry in nut_layer:
        ingrs = entry['ingredients']
        units = entry['unit']

        for ingr, unit in zip(ingrs, units):
            ingr = ingr['text'].split()[0]
            unit = unit['text']
            if ingr in ingr2unit.keys():
                ingr2unit[ingr].add(unit)
            else:
                ingr2unit[ingr] = set()
                ingr2unit[ingr].add(unit)
    
    n_unit = 0
    for k, v in ingr2unit.items():
        ingr2unit[k] = list(v)
        n_unit += len(v)
    
    n_unit = n_unit / len(ingr2unit.keys())
    print("Average # units per ingr item: ", n_unit) ## 7.5
    
    # print(ingr2unit)
    with open('/home/donghee/inversecooking/recipe1M+/ingr2unit.json', 'w') as f:
        json.dump(ingr2unit, f, indent=4)

def check_nut_ingr():
    with open("/home/donghee/inversecooking/recipe1M+/recipes_with_nutritional_info.json", 'r') as f:
        nut_layer = json.load(f)
    with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
        layer1 = json.load(f)
    
    nut_id2ingr = dict()
    units = set()
    ingrs = set()

    for entry in nut_layer:
        nut_id2ingr[entry['id']] = len(entry['ingredients'])
        for unit in entry['unit']:
            units.add(unit['text'])
        for ingr in entry['ingredients']:
            ingrs.add(ingr['text'].split()[0])
    
    id2ingr = dict()
    for entry in layer1:
        id2ingr[entry['id']] = len(entry['ingredients'])
    
    cnt = 0
    total = 0
    for id, ingr in nut_id2ingr.items():
        total += 1
        cnt += (ingr != id2ingr[id])
    
    # print(f'unmatched samples (total): {cnt} ({total})') ## cnt = 0

    # print("len of units: ", len(units))
    # print(units)
    print("# ingrs in nut_layer: ", len(ingrs))
    print(ingrs)

def check_pkl():
    args = pickle.load(open('/home/donghee/inversecooking/results/origin_inverse/im2ingr/checkpoints/args.pkl', 'rb'))
    print(args)

# single_food_counter()
# count_ids()
# count_layer2()
# count_nut_layer()
# check_nut_ingr()
check_pkl()