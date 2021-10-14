# this is the main entrypoint
# as we describe in the paper, we compute the flops over the first 100 images
# on COCO val2017, and report the average result
import torch
import time
import torchvision
import argparse

import numpy as np
import tqdm

from models import build_model
from datasets import build_dataset

from flop_count import flop_count


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--train_image_set', default='train')## add for train on sampled set, train_sampled_PER_CAT_THR_500, ...
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--sample_reg_loss', default=1e-4, type=float,
                        help="sample_reg_loss")
    parser.add_argument('--sample_topk_ratio', default=1/3., type=float)
    parser.add_argument('--score_pred_net', type=str, default='2layer-fc-256')
    parser.add_argument('--unsample_abstract_number', default=100, type=int,
                        help='unsample_abstract_number')
    parser.add_argument('--pos_embed_kproj', action='store_true',
                        help="add pos embeding for predicting unsampled aggregation attention")
    parser.add_argument('--sampler_lr_drop_epoch', default=1e5, type=int,
                        help='default is not drop')
    parser.add_argument('--reshape_param_group', action='store_true',
                        help="reshape_param_group of loaded state_dict to match with the 3 group setting")
    parser.add_argument('--notload_lr_scheduler', action='store_true',
                        help="notload_lr_scheduler")
    return parser

def get_dataset(coco_path):
    """
    Gets the COCO dataset used for computing the flops on
    """
    class DummyArgs:
        pass
    args = DummyArgs()
    args.dataset_file = "coco"
    args.coco_path = coco_path
    args.masks = False
    dataset = build_dataset(image_set='val', args=args)
    return dataset


def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()


# get the first 100 images of COCO val2017
PATH_TO_COCO = "./data/coco/"
dataset = get_dataset(PATH_TO_COCO)
images = []
for idx in range(100):
    img, t = dataset[idx]
    images.append(img)

device = torch.device('cuda')
results = {}

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

model, criterion, postprocessors = build_model(args)
model.to(device)

model_name = 'detr_resnet50'

with torch.no_grad():
    tmp = []
    tmp2 = []
    measure_scopes = ['encoder','decoder','backbone','SortSampler']
    measure_scopes_res = {k:[] for k in measure_scopes}
    for img in tqdm.tqdm(images):
        inputs = [img.to(device)]
        res = flop_count(model, (inputs,))
        [measure_scopes_res[k].append(sum(flop_count(model, (inputs,), measure_scope=k).values())) for k in measure_scopes]
        # t = measure_time(model, inputs)
        tmp.append(sum(res.values()))
        # tmp2.append(t)
results[model_name] = {'flops': fmt_res(np.array(tmp)),
                       'flops_backbone': np.mean(measure_scopes_res['backbone']),
                       'flops_encoder': np.mean(measure_scopes_res['encoder']),
                       'flops_decoder': np.mean(measure_scopes_res['decoder']),
                       'flops_sampler': np.mean(measure_scopes_res['SortSampler']),
                       }


print('=============================')
print('')
for r in results:
    print(r)
    for k, v in results[r].items():
        print(' ', k, ':', v)