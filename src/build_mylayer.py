import json
import random
import os
import urllib.request
from tqdm import tqdm
import logging
# from multiprocessing import Pool
from threading import Thread
import time
from concurrent.futures import ThreadPoolExecutor
# import asyncio
# import aiohttp
# import requests
# import time
# import wget
# import socket

# socket.setdefaulttimeout(30)


# if __name__ == '__main__':
#     log_path = '/home/donghee/inversecooking/layer_log.log'
#     logging.basicConfig(filename = log_path, level = logging.INFO)

#     with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
#         layer1 = json.load(f)

#     with open('/home/donghee/inversecooking/recipe1M/layer2.json', 'r') as f:
#         layer2 = json.load(f)

#     with open('/home/donghee/inversecooking/recipe1M+/layer2+.json', 'r') as f:
#         layer2p = json.load(f)

#     base_path = '/home/donghee/inversecooking/recipe1M/images'

#     id2partition = dict()
#     for entry in layer1:
#         id2partition[entry['id']] = entry['partition']

#     layer2_ids = [d['id'] for d in layer2] ## already matching images
#     layer2_ids = set(layer2_ids)

#     logging.info(f'original layer2 length : {len(layer2)}')

#     for entry in tqdm(layer2p):
#         if entry['id'] not in layer2_ids: ## no matching images
            
            
#             for cnt in range(10):
#                 try:
#                     i = random.randint(0, len(entry['images'])-1)
#                     img_url = entry['images'][i]['url']
#                     img_id = entry['images'][i]['id']
#                     partition = id2partition[entry['id']]
#                     img_path = os.path.join(base_path, partition, img_id[0], img_id[1], img_id[2], img_id[3], img_id)
#                     urllib.request.urlretrieve(img_url, img_path)
#                     # flag = wget.download(img_url, out = img_path, timeout = 30)
#                     break
#                 except KeyboardInterrupt:
#                     exit()
#                 except:
#                     # print("fail to download. try another one...")
#                     continue
            
#             if cnt == 9:
#                 logging.warning(f'** Failed to download image for recipe {entry["id"]}')
#                 print("** Fail to download. ID : ", entry['id'])
#             else:
#                 layer2.append({
#                     'id': entry['id'],
#                     'images':[{
#                         'id':img_id,
#                         'url':img_url,
#                     }],
#                 })

#     with open('/home/donghee/inversecooking/recipe1M/extended_layer2.json', 'x') as f:
#         json.dump(layer2, f)

#     logging.info(f'extended layer2 length : {len(layer2)}')
#     logging.info(f'total number of recipes : {len(layer1)}')
#     print("extended layer2 length: ", len(layer2))
#     print("total number of recipes: ", len(layer1))

#     logging.info(f'Done!')
#     print("Done")


# define a function to download the image and append the ID to a list
def download_image(idx, entry, id2partition, layer2, base_path, counter):

    for cnt in range(10):
        try:
            i = random.randint(0, len(entry['images'])-1)
            img_url = entry['images'][i]['url']
            img_id = entry['images'][i]['id']
            partition = id2partition[entry['id']]
            img_path = os.path.join(base_path, partition, img_id[0], img_id[1], img_id[2], img_id[3], img_id)
            urllib.request.urlretrieve(img_url, img_path)
            layer2.append({
                'id': entry['id'],
                'images':[{
                    'id':img_id,
                    'url':img_url,
                }],
            })
            break
        except KeyboardInterrupt:
            exit()
        except:
            # print("fail to download. try another one...")
            continue

    if cnt == 9:
        logging.warning(f'** Failed to download image for recipe {entry["id"]}')
        # print("** Fail to download. ID : ", entry['id'])

    else:
        # print(f"Downloaded {img_id}")
        counter[0] += 1

    if counter[0] % 10000 == 0:
        # end = time.time()
        save_layer2(layer2)
        logging.info(f'## layer2 save. index: {idx}')
        print("## layer2 save. index: ", idx)
        # try:
        #     t = end -start
        #     logging.info(f'## 10000 download time: {t}')
        #     print("# 10000 download time: ", t)
        # except:
        #     print("First 10000 iteration")
        # start = time.time()

# def download_images(idx, entry, id2partition, layer2, counter):

#     download_image(idx, entry, id2partition, layer2, counter) 
#     # thread = Thread(target=download_image, args=(idx, entry, id2partition, layer2, counter))
#     # thread.start()
#     # thread.join()


def save_layer2(layer2):
    with open('/home/donghee/inversecooking/recipe1M/added_layer2.json', 'w') as f:
        json.dump(layer2, f, indent=4)


def download_1M():
    log_path = '/home/donghee/inversecooking/layer_log.log'
    logging.basicConfig(filename = log_path, level = logging.INFO)

    # load data from json files
    with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
        layer1 = json.load(f)

    # with open('/home/donghee/inversecooking/recipe1M/layer2.json', 'r') as f:
    #     layer2 = json.load(f)

    with open('/home/donghee/inversecooking/recipe1M+/layer2+.json', 'r') as f:
        layer2p = json.load(f)

    layer2 = []
    # set up base paths and mappings
    base_path = '/home/donghee/inversecooking/recipe1M/images'

    id2partition = dict()
    for entry in layer1:
        id2partition[entry['id']] = entry['partition']

    layer2_ids = set(d['id'] for d in layer2) ## set of matching images

    logging.info(f'original layer2 length : {len(layer2)}')

    # download images and append IDs to layer2 list using multithreading
    # max_workers = 16
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = []
    #     for idx, entry in enumerate(tqdm(layer2p)):
    #         if entry['id'] not in layer2_ids: ## no matching images
    #             futures.append(executor.submit(download_images, idx, entry, id2partition, layer2, counter))
    #             # thread = Thread(target=download_images, args=(idx, entry, id2partition, layer2, counter))
    #             # threads.append(thread)
    #         if len(futures) == max_workers:
    #             for future in futures:
    #                 future.result()
    #             futures = []
            
    #     for future in futures:
    #         future.result()
    last_id = 'c550050adb'
    for i, entry in enumerate(layer2p):
        if entry['id'] == last_id:
            start_index = i
            break

    threads = []
    counter = [0]
    max_threads = 100

    for idx, entry in enumerate(tqdm(layer2p[start_index:])):
        if entry['id'] not in layer2_ids: ## no matching images
            thread = Thread(target=download_image, args=(idx, entry, id2partition, layer2, base_path, counter))
            threads.append(thread)
            # if len(threads) == max_threads:
            #     # start the threads
            #     for thread in threads:
            #         thread.start()
            #     # wait for all threads to finish
            #     for thread in threads:
            #         thread.join(timeout=10)
            #         if thread.is_alive():
            #             logging.warning(f'Thread {thread.name} cannot be completed within 10 seconds and is killed')
            #             thread.kill_received = True
            #     threads = []
            
    # start the threads
    for thread in tqdm(threads):
        try:
            thread.start()
        except:
            print("cannot start thread anymore")
            for t in threads:
                t.join(timeout=10)
                if t.is_alive():
                    logging.warning(f'Thread {thread.name} cannot be completed within 10 seconds and is killed')
                    t.kill_received = True
            save_layer2(layer2)

    # wait for all threads to finish
    for thread in threads:
        thread.join(timeout=10)
        if thread.is_alive():
            logging.warning(f'Thread {thread.name} cannot be completed within 10 seconds and is killed')
            thread.kill_received = True


    # save layer2 to a json file
    save_layer2(layer2)

    logging.info(f'extended layer2 length : {len(layer2)}')
    logging.info(f'total number of recipes : {len(layer1)}')
    print("extended layer2 length: ", len(layer2))
    print("total number of recipes: ", len(layer1))

    logging.info(f'Done!')
    print("Done")

def build_json():

    with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
        layer1 = json.load(f)
    with open('/home/donghee/inversecooking/recipe1M/layer2.json', 'r') as f:
        layer2 = json.load(f)
    with open('/home/donghee/inversecooking/recipe1M+/layer2+.json', 'r') as f:
        layer2p = json.load(f)
    
    imgid2idx = {}
    for i, entry in enumerate(tqdm(layer2p)):
        for j, image in enumerate(entry['images']):
            imgid2idx[image['id']] = {'id': entry['id'], 'url': image['url']}
    
    layer2_ids = set([entry['id'] for entry in layer2])
    original_len = len(layer2)

    cnt = 0
    for entry in layer2: ## layer2 total images
        cnt += len(entry['images'])
    print("layer2 total number of images: ", cnt)

    print("len of layer1: ", len(layer1))
    print("origianl len of layer2: ", len(layer2))
    print("layer2 ids: ", len(layer2_ids))
    assert len(layer2) == len(layer2_ids)
    # Set the root directory
    root_dir = '/home/donghee/inversecooking/recipe1M/images/'

    removed_images = []
    # Loop through the train, test, and val directories
    for split_dir in ['train', 'test', 'val']:
        # Set the directory for the current split
        split_dir = os.path.join(root_dir, split_dir)
        # Walk through all the subdirectories and files in the split directory
        for subdir, _, files in tqdm(os.walk(split_dir)):
            # Loop through all the files in the current directory
            for file in files:
                # Check if the file is a jpg file
                img_id = file
                try:
                    id = imgid2idx[img_id]['id']
                except:
                    print("the image is removed in layer2p")
                    removed_images.append(img_id)
                    continue
                img_url = imgid2idx[img_id]['url']
                if file.endswith('.jpg') and id not in layer2_ids: ## 다운됐는데 아직 레이어에 추가되지 않은 케이스
                    layer2.append({
                    'id': id,
                    'images':[{
                        'id':img_id,
                        'url':img_url,
                    }],
                    })
        
        updated_len = len(layer2) - original_len
        print("# {} added: {}".format(split_dir, updated_len)) ## train: 1098127, test: 233276, val: 231184
        original_len = len(layer2)

    print("** Total number of ids: ", len(layer2))
    
    cnt = 0
    for entry in tqdm(layer2):
        cnt += len(entry['images'])
    print("** Total number of images in extended json: ", cnt)
    ## total number of images (train/test/val): 1562587
    print("Removed images (from layer2p): ", removed_images)
    with open('/home/donghee/inversecooking/recipe1M/extended_layer2.json', 'w') as f:
        json.dump(layer2, f)
    
    print("DONE!")


def check_json():
    with open('/home/donghee/inversecooking/recipe1M/extended_layer2.json', 'r') as f:
        extended_layer2 = json.load(f)
    with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
        layer1 = json.load(f)

    print("len layer1: ", len(layer1))
    print("len extended layer: ", len(extended_layer2))

    extended_ids = set([entry['id'] for entry in extended_layer2])
    layer1_ids = set([entry['id'] for entry in layer1])

    print("len extended ids: ", len(extended_ids)) ## id 중복으로 들어간 것들이 있음!! (3만개 정도) // 총 5만개 non-matching images.
    print("len layer1 ids: ", len(layer1_ids))

    non_matching = layer1_ids - extended_ids
    wrong = extended_ids - layer1_ids ## should be None

    print("len non_matching: ", len(non_matching))
    print("len wrong: ", len(wrong))


def remove_duplicated_id():

    with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
        layer1 = json.load(f)
    with open('/home/donghee/inversecooking/recipe1M/extended_layer2.json', 'r') as f:
        extended_layer2 = json.load(f)
    
    ids = set()
    duplicated_ids = []
    duplicated_idx = [] ## 나중에 pop 하기 위한..
    id2idx = dict()

    for i, entry in enumerate(tqdm(extended_layer2)):
        if entry['id'] in ids:
            duplicated_ids.append(entry['id'])
            idx = id2idx[entry['id']]
            duplicated_idx.append(idx)
            duplicated_entry = extended_layer2[idx]
            assert duplicated_entry['id'] == entry['id']
            image_ids = set([image['id'] for image in entry['images']])
            for image in duplicated_entry['images']:
                if image['id'] not in image_ids:
                    entry['images'].append(image)
        
        ids.add(entry['id'])
        id2idx[entry['id']] = i
    

    print("len layer1: ", len(layer1)) ## 1,029,720
    print("original extended layer2 len: ", len(extended_layer2)) ## 1,077,316
    print("len of duplicated index: ", len(duplicated_idx)) ## 100,623

    assert len(duplicated_idx) == len(set(duplicated_idx))
    duplicated_idx = set(duplicated_idx)
    new_layer2 = [element for i, element in enumerate(extended_layer2) if i not in duplicated_idx] ## 100,623개 제거
    print("len of new_layer2: ", len(new_layer2)) ## 976,693 ## all unique id (good!) ## all valid !
    ## non-matching ids = 53,027
    with open('/home/donghee/inversecooking/recipe1M/layer2_1M.json', 'w') as f:
        json.dump(new_layer2, f, indent=4) ## total number of images = 1,562,092

def download_nonmatching(): ## 못했음!!

    def download_image(img_url, img_path):
        urllib.request.urlretrieve(img_url, img_path)
    
    with open('/home/donghee/inversecooking/recipe1M/layer2_1M.json', 'r') as f:
        new_layer2 = json.load(f)
    with open('/home/donghee/inversecooking/recipe1M+/layer2+.json', 'r') as f:
        layer2p = json.load(f)
    
    layer2p_ids = set([entry['id'] for entry in layer2p if len(entry['images']) != 0])
    print("len layer2p: ", len(layer2p_ids)) ## 1,016,932
    # new_layer2_noimage = [idx for idx, entry in enumerate(new_layer2) if len(entry['images']) == 0] ## no image index ## 0
    new_layer2_ids = set([entry['id'] for entry in new_layer2])
    original_new_layer2_len = len(new_layer2)

    with open('/home/donghee/inversecooking/recipe1M/layer1.json', 'r') as f:
        layer1 = json.load(f)
    id2partition = dict()
    layer1_ids = set()
    for entry in layer1:
        id2partition[entry['id']] = entry['partition']
        layer1_ids.add(entry['id'])

    base_path = '/home/donghee/inversecooking/recipe1M/images'
    fail_cnt = 0
    success_cnt = 0
    for i, entry in enumerate(tqdm(layer2p)):
        if i == 396910 or i == 398048:
            continue
        if entry['id'] not in new_layer2_ids:
            for image in entry['images']: ## 차례로..
                img_url = image['url']
                img_id = image['id']
                partition = id2partition[entry['id']]
                img_path = os.path.join(base_path, partition, img_id[0], img_id[1], img_id[2], img_id[3], img_id)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(download_image, img_url, img_path)
                    try:
                        future.result(timeout=10)
                        new_layer2.append({
                        'id': entry['id'],
                        'images':[{
                            'id':img_id,
                            'url':img_url,
                        }],
                        })
                        success_cnt += 1
                        break
                    except TimeoutError:
                        print("** time out. try next one...")
                        fail_cnt += 1
                        continue
                    except:
                        print("fail to download, try next one...")
                        fail_cnt += 1
                        continue
    
    print("success: ", success_cnt)
    print("fail: ", fail_cnt)

    print("** len new_layer2: ", len(new_layer2))
    print("original new_layer2: ", original_new_layer2_len)
    diff = original_new_layer2_len-len(new_layer2)
    print("diff: ", diff)
    assert diff == success_cnt
    ## 1,029,720 - layer1 (total number of recipes) ## layer2p 1,016,932
    extended_new_layer2_ids = set([entry['id'] for entry in new_layer2])
    assert len(extended_new_layer2_ids) == len(new_layer2)

    no_images = layer1_ids - extended_new_layer2_ids
    print("len of no images: ", len(no_images))
    wrong = extended_new_layer2_ids - layer1_ids
    assert len(wrong) == 0

    with open('/home/donghee/inversecooking/recipe1M/layer2_1M_extended.json', 'w') as f:
        json.dump(new_layer2, f, indent=4)

def clean_json():
    ## layer2_1M clean 해서 layer2_1M_clean.json
    ## load_error 제거 (load_error - lmdb 실패)
    with open('/home/donghee/inversecooking/recipe1M/layer2_1M.json', 'r') as f:
        layer2_1M = json.load(f)
    
    with open('/home/donghee/inversecooking/recipe1M/load_error.json', 'r') as f:
        load_error = json.load(f)
    
    load_error = set(load_error) ## 13801
    
    remove_cnt = 0
    for entry in layer2_1M:
        images = entry['images']
        for i, image in enumerate(images):
            if image['id'] in load_error:
                entry['images'].pop(i)
                remove_cnt += 1
    

    # assert len(load_error) == remove_cnt
    print("remove_cnt: ", remove_cnt)

    with open('/home/donghee/inversecooking/recipe1M/layer2_1M_clean.json', 'w') as f:
        json.dump(layer2_1M, f, indent=4)
    
    print("Done!")





# build_json()
# check_json()
# remove_duplicated_id()
# download_nonmatching()
clean_json()