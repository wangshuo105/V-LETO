import pandas as pd
import torch
from lib.data_example import *
from lib.options import args_parser
from random import choice
# from lib.models.models import *
from lib.models.models_cifar import *
# from lib.models.resnet_cifar import *
from lib.models.resnet_mnist import *
# from fed.active_party_cifar import active_party_cifar
# from fed.passive_cifar import passive_party_cifar
import openpyxl
from torch.utils.data import DataLoader
import pandas as pd
import os
import time
import copy
import random
import numpy as np
from skimage import io,img_as_ubyte
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from datetime import datetime
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# from openTSNE import TSNE
import torch.nn as nn
import logging
from sklearn.utils import resample
import time
from sklearn.preprocessing import normalize



def agg_func(protos):

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        print(f"f folder {folder_name} exist")

def setup_logger(args):
  
    log_dir = f'./logs_logger/{args.datasets}'
    create_folder(log_dir)

    logger = logging.getLogger("VFCL_Logger")
    logger.setLevel(logging.INFO)


    logger.info(args)
    

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)


    log_filename = os.path.join(log_dir, f'VFCL_class_{args.datasets}_{args.model}_{args.global_epoch}_{args.weight_type}_task_{args.task}_{args.task_type}_{args.num_user}_{args.k_0}_{args.alpha}_train_{time.strftime("%Y_%m_%d_%H_%M_%S")}.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)


    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def distance_prototype(pre_prototype,current_prototype):
    euclidean_distance ={}
    for key in pre_prototype:
        if key in current_prototype:
            euclidean_distance[key] = torch.norm(pre_prototype[key] - current_prototype[key])
    
    total_distance = sum(euclidean_distance.values())
    average_distance = total_distance / len(euclidean_distance) 
    return average_distance
        
def filter_by_class(dataset, classes):
    indices = [i for i, (img, label) in enumerate(dataset) if label in classes]
    return torch.utils.data.Subset(dataset, indices)

def agg_func(protos):

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def agg_func_error(protos):

    average_protos = {}
    errors = {}

    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = torch.zeros_like(proto_list[0].data)
            for i in proto_list:
                proto += i.data
            average_proto = proto / len(proto_list)
            average_protos[label] = average_proto

            error = torch.zeros_like(proto_list[0].data)
            for i in proto_list:
                error += (i.data - average_proto).abs()
            errors[label] = error / len(proto_list)
        else:
            average_protos[label] = proto_list[0]
            errors[label] = torch.zeros_like(proto_list[0].data)

    return average_protos, errors

def find_closest_label(all_inter, current_prototype):
    closest_labels = []
    for inter in all_inter:
        min_distance = float('inf')
        closest_label = None
        for label, proto in current_prototype.items():
            distance = torch.norm(inter - proto)
            if distance < min_distance:
                min_distance = distance
                closest_label = label
        closest_labels.append(closest_label)
    return torch.tensor(closest_labels)

def all_prototype_current_compute(global_prototype, union_prototype, euclidean_distance):
    old_prototype_current = {}
    for key in union_prototype:
        if key in global_prototype:
            new_value = global_prototype[key] + args.r * euclidean_distance
            old_prototype_current[key] = new_value
        else:
            old_prototype_current[key] = union_prototype[key]
    return old_prototype_current




def select_important_parameters(fisher_information, threshold=0.01):
    important_params = {}
    for name, fisher_value in fisher_information.items():
        important_params[name] = (fisher_value > threshold).float()
    return important_params




def proto_aug_compute(global_prototype,t,number_data):
    proto_aug = []
    proto_aug_label = []
    
    

    index = [k for k, v in global_prototype.items() if torch.sum(v) != 0]
    for key in index:  
        for _ in range(number_data):
            original_proto = global_prototype[key]
            noise = torch.tensor(np.random.normal(0, 1, original_proto.size()) * args.radius).float()
            noise = noise.to(device)

            augmented_proto = original_proto + noise
            proto_aug.append(augmented_proto)
            

            proto_aug_label.append(list(global_prototype.keys()).index(key))



    proto_aug = torch.stack(proto_aug).to(torch.float32)
    proto_aug_label = torch.tensor(proto_aug_label).long()
    return proto_aug, proto_aug_label


def temperature_squared_mse_loss(outputs_student, outputs_teacher, temperature=1.0):

    soft_logits_student = F.softmax(outputs_student / temperature, dim=1)
    soft_logits_teacher = F.softmax(outputs_teacher / temperature, dim=1)
    

    loss = F.mse_loss(soft_logits_student, soft_logits_teacher)
    return loss

def initialize_with_noise(old_model, noise_scale=0.01):
    for param in old_model.parameters():
        noise = torch.randn_like(param) * noise_scale
        param.data.add_(noise)
        
        

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        cos_sim = F.cosine_similarity(output1, output2)
 
        cos_distance = 1 - cos_sim

        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 2))
        return loss_contrastive
