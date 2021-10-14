import copy
import json
import os
from collections import defaultdict

from random import sample
# import random
import numpy as np
import tqdm

import matplotlib.pyplot as plt
if __name__ == "__main__":
    ann_file = './data/coco/annotations/instances_train2017.json'
    PER_CAT_THR = 1000
    output_filename = './data/coco/annotations/instances_train2017_sampled_PER_CAT_THR_{}.json'.format(PER_CAT_THR)

    with open(ann_file, "r") as f:
        dataset = json.load(f)

    catToImgs = defaultdict(list)
    for ann in dataset['annotations']:
        catToImgs[ann['category_id']].append(ann['image_id'])

    ## remove duplicate imgs
    for cat_id in catToImgs.keys():
        catToImgs[cat_id] = list(set(catToImgs[cat_id]))

    per_cat_img_number = [len(catToImgs[cat_id]) for cat_id in catToImgs.keys()]
    sorting_order_imgnumber = np.argsort(per_cat_img_number).tolist()
    sorted_catid = [list(catToImgs.keys())[i] for i in sorting_order_imgnumber]

    catToImgs_list = [{cat_id:catToImgs[cat_id]}for cat_id in catToImgs.keys()]
    catToImgs_sampled = copy.deepcopy(catToImgs)
    sampled_img_ids = []

    for cat_id in tqdm.tqdm(sorted_catid):## starting from cat with least imgs
        if len(catToImgs[cat_id])>PER_CAT_THR: # only sample categories with more than 2000 training imgs
            in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id in sampled_img_ids]
            not_in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id not in sampled_img_ids]

            catToImgs_sampled[cat_id] = in_sampled + sample(not_in_sampled, PER_CAT_THR-len(in_sampled)) if len(in_sampled)<PER_CAT_THR else in_sampled
        sampled_img_ids+=catToImgs_sampled[cat_id]

    # catToImgs_list = [{cat_id:catToImgs[cat_id]}for cat_id in catToImgs.keys()]
    # catToImgs_sampled_2000 = copy.deepcopy(catToImgs)
    # PER_CAT_THR = 2000
    # sampled_img_ids = []
    #
    # for cat_id in tqdm.tqdm(sorted_catid):## starting from cat with least imgs
    #     if len(catToImgs[cat_id])>PER_CAT_THR: # only sample categories with more than 2000 training imgs
    #         in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id in sampled_img_ids]
    #         not_in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id not in sampled_img_ids]
    #
    #         catToImgs_sampled_2000[cat_id] = in_sampled + sample(not_in_sampled, PER_CAT_THR-len(in_sampled)) if len(in_sampled)<PER_CAT_THR else in_sampled
    #     sampled_img_ids+=catToImgs_sampled_2000[cat_id]
    # img_number_per_class_2000 = [len(catToImgs_sampled_2000[i]) for i in sorted_catid]
    #
    #
    # catToImgs_sampled_1000 = copy.deepcopy(catToImgs)
    # PER_CAT_THR = 1000
    # sampled_img_ids = []
    #
    # for cat_id in tqdm.tqdm(sorted_catid):## starting from cat with least imgs
    #     if len(catToImgs[cat_id])>PER_CAT_THR: # only sample categories with more than 2000 training imgs
    #         in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id in sampled_img_ids]
    #         not_in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id not in sampled_img_ids]
    #
    #         catToImgs_sampled_1000[cat_id] = in_sampled + sample(not_in_sampled, PER_CAT_THR-len(in_sampled)) if len(in_sampled)<PER_CAT_THR else in_sampled
    #     sampled_img_ids+=catToImgs_sampled_1000[cat_id]
    # img_number_per_class_1000 = [len(catToImgs_sampled_1000[i]) for i in sorted_catid]
    #
    #
    # catToImgs_sampled_500 = copy.deepcopy(catToImgs)
    # PER_CAT_THR = 500
    # sampled_img_ids = []
    #
    # for cat_id in tqdm.tqdm(sorted_catid):## starting from cat with least imgs
    #     if len(catToImgs[cat_id])>PER_CAT_THR: # only sample categories with more than 2000 training imgs
    #         in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id in sampled_img_ids]
    #         not_in_sampled = [img_id for img_id in catToImgs[cat_id] if img_id not in sampled_img_ids]
    #
    #         catToImgs_sampled_500[cat_id] = in_sampled + sample(not_in_sampled, PER_CAT_THR-len(in_sampled)) if len(in_sampled)<PER_CAT_THR else in_sampled
    #     sampled_img_ids+=catToImgs_sampled_500[cat_id]
    # img_number_per_class_500 = [len(catToImgs_sampled_500[i]) for i in sorted_catid]

    # img_number_per_class_500 = [len(catToImgs_sampled_500[i]) for i in sorted_catid]
    # img_number_per_class_1000 = [len(catToImgs_sampled_1000[i]) for i in sorted_catid]
    # img_number_per_class_2000 = [len(catToImgs_sampled_2000[i]) for i in sorted_catid]
    # img_number_per_class_org = [len(catToImgs[i]) for i in sorted_catid]
    # index = np.arange(len(img_number_per_class))
    # plt.bar(index - 0.45, img_number_per_class_org, width=0.3, label='original')
    # plt.bar(index - 0.15, img_number_per_class_2000, width=0.3, label='per class 2000')
    # plt.bar(index + 0.15, img_number_per_class_1000, width=0.3, label='per class 1000')
    # plt.bar(index + 0.45, img_number_per_class_500, width=0.3, label='per class 500')
    # plt.yscale('log')
    # plt.xlabel('sorted class index')
    # plt.ylabel('per class image nubmer')
    # plt.legend()
    # plt.show()

    temp = []
    for cat_id in catToImgs.keys():
        temp += catToImgs_sampled[cat_id]
    final_sampled_imgs = set(temp)
    print(len(final_sampled_imgs))

    ## filter imgs
    new_image_list = []
    for image in dataset['images']:
        if image['id'] in final_sampled_imgs:
            new_image_list.append(image)
    dataset['images'] = new_image_list

    ## filter anns
    new_anns_list = []
    for ann in dataset['annotations']:
        if ann['image_id'] in final_sampled_imgs:
            new_anns_list.append(ann)
    dataset['annotations'] = new_anns_list


    with open(output_filename, "w") as f:
        json.dump(dataset, f)
    print("{} is COCOfied and stored in {}.".format(ann_file, output_filename.format(PER_CAT_THR)))


    print('df')