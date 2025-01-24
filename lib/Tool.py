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
    """
    Returns the average of the weights.
    """

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
        #print(f"Fold{folder_name} finish")
    except FileExistsError:
        print(f"f folder {folder_name} exist")

def setup_logger(args):
    """
    设置日志记录器，输出日志到控制台和文件
    """
    # Ensure the 'logs' directory exists
    log_dir = f'./logs_logger/{args.datasets}'
    create_folder(log_dir)

    logger = logging.getLogger("VFCL_Logger")
    logger.setLevel(logging.INFO)


    logger.info(args)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器，文件路径包括时间戳
    log_filename = os.path.join(log_dir, f'VFCL_class_{args.datasets}_{args.model}_{args.global_epoch}_{args.weight_type}_task_{args.task}_{args.task_type}_{args.num_user}_{args.k_0}_{args.alpha}_train_{time.strftime("%Y_%m_%d_%H_%M_%S")}.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
# def load_model(model, model_path):
#     # 加载保存的 state_dict
#     model.load_state_dict(torch.load(model_path))
#     # model.eval()  # 切换到评估模式
#     return model

class FilteredMNIST(datasets.MNIST):
    def __init__(self, *args, labels=(0, 1), **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        
        # 筛选出指定标签的样本
        self.indices = [i for i in range(len(self)) if self.targets[i].item() in labels]
        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]
    
    def __getitem__(self, index):
        # 返回筛选后的样本
        return self.data[index], self.targets[index]

    def __len__(self):
        # 返回筛选后的样本数量
        return len(self.data)

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
    """
    Returns the average of the weights.
    """

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
    """
    Returns the average of the weights and the error for each label.
    """
    average_protos = {}
    errors = {}

    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = torch.zeros_like(proto_list[0].data)
            for i in proto_list:
                proto += i.data
            average_proto = proto / len(proto_list)
            average_protos[label] = average_proto

            # 计算误差
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
            # 计算新的值：global_prototype[key] + A * euclidean_distance[key]
            new_value = global_prototype[key] + args.r * euclidean_distance
            # 将键和值添加到 old_prototype_current 中
            old_prototype_current[key] = new_value
        else:
            old_prototype_current[key] = union_prototype[key]
    return old_prototype_current




def select_important_parameters(fisher_information, threshold=0.01):
    important_params = {}
    for name, fisher_value in fisher_information.items():
        # 选择费舍尔信息值大于阈值的参数
        important_params[name] = (fisher_value > threshold).float()
    return important_params




def proto_aug_compute(global_prototype,t,number_data):
    proto_aug = []
    proto_aug_label = []
    
    
    # 从字典中获取非空原型的键
    index = [k for k, v in global_prototype.items() if torch.sum(v) != 0]
    for key in index:  # 遍历 index 中的所有元素
        for _ in range(number_data):
            original_proto = global_prototype[key]
            # 添加噪声以生成增强原型
            noise = torch.tensor(np.random.normal(0, 1, original_proto.size()) * args.radius).float()
            noise = noise.to(device)
            
            # 生成增强后的原型
            augmented_proto = original_proto + noise
            proto_aug.append(augmented_proto)
            
            # 添加对应的标签
            proto_aug_label.append(list(global_prototype.keys()).index(key))
        # print("augmented_proto",augmented_proto)
        # print("augmented_label",list(global_prototype.keys()).index(key))

    # 将增强的原型和标签转换为张量
    proto_aug = torch.stack(proto_aug).to(torch.float32)
    proto_aug_label = torch.tensor(proto_aug_label).long()
    return proto_aug, proto_aug_label


def temperature_squared_mse_loss(outputs_student, outputs_teacher, temperature=1.0):
    """
    计算使用温度参数调整后的学生和教师模型输出之间的均方误差损失。
    outputs_student: 学生模型的原始输出 logits。
    outputs_teacher: 教师模型的原始输出 logits。
    temperature: 用于软化 softmax 的温度参数。
    """
    # 使用温度调整 softmax
    soft_logits_student = F.softmax(outputs_student / temperature, dim=1)
    soft_logits_teacher = F.softmax(outputs_teacher / temperature, dim=1)
    
    # 计算两者的 MSE
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
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(output1, output2)
        # 将余弦相似度转换为距离度量
        cos_distance = 1 - cos_sim
        # 计算对比损失
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 2))
        return loss_contrastive



def Scatter_plot_list(all_inter_list, label_list, epoch_prototype, old_prototype_current, t, args, folder_name):
    # 将 label_list 从 GPU 移动到 CPU 并转换为 NumPy 数组
    label_list = label_list.cpu().numpy()
    
    # 获取唯一的标签值
    unique_labels = np.unique(label_list)
    num_classes = len(unique_labels)
    
    # 设置默认的 colors 和 class_names
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00',
              '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#006400', '#FFC0CB']
    class_names = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    # 创建文件夹
    output_folder = f"save/scatter_plots/{folder_name}"
    os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在就创建

    plt.figure(figsize=(10, 8))
    
    # 对每个标签进行采样
    sampled_inter_list = []
    sampled_label_list = []
    # unique_labels = np.unique(label_list)
    for label in unique_labels:
        indices = np.where(label_list == label)[0]
        # n_samples = min(4000, len(indices))
        # replace = len(indices) < 4000
        sampled_indices = resample(indices, n_samples=4000, replace=True, random_state=42)
        sampled_inter_list.append(all_inter_list[sampled_indices].detach().cpu().numpy())
        sampled_label_list.append(label_list[sampled_indices])

    sampled_inter_list = np.vstack(sampled_inter_list)
    sampled_label_list = np.hstack(sampled_label_list)

    # 使用 L2 范数进行归一化
    sampled_inter_list = normalize(sampled_inter_list, norm='l2')

    # 将 epoch_prototype 转换为 NumPy 数组并进行归一化
    prototype_list = np.array([epoch_prototype[label].cpu().detach().numpy() for label in sorted(epoch_prototype.keys())])
    prototype_list = normalize(prototype_list, norm='l2')
    old_prototype_current_list = np.array([old_prototype_current[label].cpu().detach().numpy() for label in sorted(old_prototype_current.keys())])
    old_prototype_current_list = normalize(old_prototype_current_list, norm='l2')

    # 合并数据进行 t-SNE 降维
    combined_data = np.vstack([sampled_inter_list, prototype_list, old_prototype_current_list])
    
    # 确保 perplexity 小于样本数量
    perplexity = min(30, len(combined_data) - 1)
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_jobs=-1)
    combined_data_2d = tsne.fit_transform(combined_data)

    len_sampled = len(sampled_inter_list)
    len_prototype = len(prototype_list)
    len_old_prototype = len(old_prototype_current_list)

    sampled_inter1_2d = combined_data_2d[:len_sampled]
    prototype_2d = combined_data_2d[len_sampled:len_sampled + len_prototype]
    old_prototype_current_2d = combined_data_2d[len_sampled + len_prototype:]
    

    # 绘制采样后的散点图
    for label in unique_labels:
        indices = (sampled_label_list == label)  # 获取当前标签的索引
        plt.scatter(
            sampled_inter1_2d[indices, 0],  # t-SNE 第一维
            sampled_inter1_2d[indices, 1],  # t-SNE 第二维
            label=class_names[label],  # 使用类别名称作为标签
            alpha=0.5,
            s=10,  # 点的大小
            color=colors[int(label) % len(colors)]  # 确保索引是整数
        )

    # 绘制 epoch_prototype 的点并按照标签顺序输出图例
    for i, label in enumerate(sorted(epoch_prototype.keys())):
        plt.scatter(
            prototype_2d[i, 0], prototype_2d[i, 1], 
            # label=f"Prototype {class_names[label]}", 
            marker='s', 
            color=colors[int(label) % len(colors)], 
            edgecolors='black',  # 设置边框颜色为黑色
            s=120
        )
    if t !=0:
        for i, label in enumerate(sorted(old_prototype_current.keys())):
            plt.scatter(
                old_prototype_current_2d[i, 0], old_prototype_current_2d[i, 1], 
                # label=f"Old Prototype {class_names[label]}", 
                marker='s', 
                color=colors[int(label) % len(colors)], 
                edgecolors='black',  # 设置边框颜色为黑色
                s=120
            )

    plt.legend(fontsize=20,loc='best')
    # plt.title()
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"scatter_plot_class_{args.datasets}_{t}_{args.model}_{args.global_epoch}_{timestamp}.pdf"))
    plt.close()

# def Scatter_plot_list(all_inter_list, label_list, epoch_prototype, t, args, folder_name):
#     # 设置默认的 colors 和 class_names
#     label_list = label_list.cpu().numpy()
#     unique_labels = np.unique(label_list)
#     print("unique_labels",unique_labels)
#     num_classes = len(unique_labels)
    
#     colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00',
#         '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#006400', '#FFC0CB']
#     class_names = {
#         0: 'T-shirt/top',
#         1: 'Trouser',
#         2: 'Pullover',
#         3: 'Dress',
#         4: 'Coat',
#         5: 'Sandal',
#         6: 'Shirt',
#         7: 'Sneaker',
#         8: 'Bag',
#         9: 'Ankle boot'
#     }

#     # 创建文件夹
#     output_folder = f"save/scatter_plots/{folder_name}"
#     os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在就创建

#     plt.figure(figsize=(10, 8))
    
#     # added_labels = [False] * num_classes
    
#     # 对每个标签进行采样
#     sampled_inter_list = []
#     sampled_label_list = []

#     for label in unique_labels:
#         indices = np.where(label_list == label)[0]
#         n_samples = min(1000, len(indices))
#         replace = len(indices) < 1000
#         sampled_indices = resample(indices, n_samples=n_samples, replace=replace, random_state=42)
#         sampled_inter_list.append(all_inter_list[sampled_indices].detach().cpu().numpy())
#         sampled_label_list.append(label_list[sampled_indices])

#     sampled_inter_list = np.vstack(sampled_inter_list)
#     sampled_label_list = np.hstack(sampled_label_list)

#     sampled_inter_list = normalize(sampled_inter_list, norm='l2')
#     # 对采样后的数据进行标准化
#     # scaler = StandardScaler()
#     # sampled_inter_list = scaler.fit_transform(sampled_inter_list)
    
#     # sampled_inter_list = (sampled_inter_list - sampled_inter_list.min(axis=0)) / (sampled_inter_list.max(axis=0) - sampled_inter_list.min(axis=0))
#     prototype_list = np.array([epoch_prototype[label].cpu().detach().numpy() for label in sorted(epoch_prototype.keys())])
#     prototype_list = normalize(prototype_list, norm='l2')
    
    
#     # t-SNE 降维
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
#     sampled_inter1_2d = tsne.fit_transform(sampled_inter_list)
    
#     prototype_2d = tsne.fit_transform(prototype_list)
    
#     # sampled_inter_list = normalize(sampled_inter1_2d, norm='l2')
    
#     # sampled_inter1_2d, prototype_2d = sampled_inter1_2d[:len(sampled_inter_list)], sampled_inter1_2d[len(sampled_inter_list):]
#     # sampled_inter1_2d = (sampled_inter1_2d - sampled_inter1_2d.min(axis=0)) / (sampled_inter1_2d.max(axis=0) - sampled_inter1_2d.min(axis=0))
#     # sampled_inter1_2d = 2 * sampled_inter1_2d - 1

#     # 绘制采样后的散点图
#     for label in unique_labels:
#         indices = (sampled_label_list == label)  # 获取当前标签的索引
#         plt.scatter(
#             sampled_inter1_2d[indices, 0],  # t-SNE 第一维
#             sampled_inter1_2d[indices, 1],  # t-SNE 第二维
#             label=class_names[label],  # 使用类别名称作为标签
#             alpha=0.5,
#             s=10,  # 点的大小
#             color=colors[int(label) % len(colors)]  # 确保索引是整数
#         )

#     # 按标签排序
#     sorted_labels = sorted(epoch_prototype.keys())

#     # 绘制 epoch_prototype 的点并按照标签顺序输出图例
#     for i, label in enumerate(sorted(epoch_prototype.keys())):
#         # prototype = epoch_prototype[label].cpu().detach().numpy()
#         plt.scatter(
#             prototype_2d[i, 0], prototype_2d[i, 1], 
#             label=f"Prototype {class_names[label]}", 
#             marker='s', 
#             color=colors[int(label) % len(colors)], 
#             edgecolors='black'  # 设置边框颜色为黑色
#         )
#     # for label in sorted_labels:
#     #     prototype = epoch_prototype[label].cpu().detach().numpy()
#     #     print(f"Prototype shape: {prototype.shape},{prototype.ndim}")
#     #     if prototype.ndim == 1:
#     #         plt.scatter(
#     #             prototype[0], prototype[1], 
#     #             label=f"Prototype {class_names[label]}", 
#     #             marker='s', 
#     #             color=colors[int(label) % len(colors)], 
#     #             edgecolors='black'  # 设置边框颜色为黑色
#     #         )
#         # if prototype.ndim == 1:
#         #     plt.scatter(prototype[0], prototype[1], label=f"Prototype {class_names[label]}", marker='s', color=colors[int(label) % len(colors)])

    
#     plt.legend(title="Labels", loc='best')
#     plt.title("t-SNE visualization of embeddings and prototypes")
#     plt.xlabel("")
#     plt.ylabel("")
#     plt.grid(True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     plt.savefig(os.path.join(output_folder, f"scatter_plot_class_{args.datasets}_{t}_{args.model}_{args.global_epoch}_{timestamp}.png"))
#     plt.close()
    # plt.legend()
    # plt.show()




# def Scatter_plot_list(all_inter_list, label_list, epoch_prototype, t, args, folder_name): 
#     num_classes = 10   # 类别数
#     print(all_inter_list.shape)
#     print(label_list.shape)
#     all_inter_list = all_inter_list.cpu().detach().numpy()
#     label_list = label_list.cpu().detach().numpy()
#     labels_tensor = torch.tensor(label_list)
    
#     # 获取唯一的标签值
#     unique_labels = torch.unique(labels_tensor)
    
#     # 输出可区分的标签值
#     print("可区分的标签值:", unique_labels.tolist())
#     label_mapping = {
#         0: 'T-shirt/top',
#         1: 'Trouser',
#         2: 'Pullover',
#         3: 'Dress',
#         4: 'Coat',
#         5: 'Sandal',
#         6: 'Shirt',
#         7: 'Sneaker',
#         8: 'Bag',
#         9: 'Ankle boot'
#     }
#     # 随机抽取一部分数据进行绘制
#     sample_size = 10000  # 你可以根据需要调整这个值
#     if all_inter_list.shape[0] > sample_size:
#         indices = np.random.choice(all_inter_list.shape[0], sample_size, replace=False)
#         all_inter_list = all_inter_list[indices]
#         label_list = label_list[indices]
    
#     if all_inter_list.ndim > 2:
#         all_inter_list = all_inter_list.reshape(all_inter_list.shape[0], -1)
#     print("all_inter_list.shape",all_inter_list.shape)
#     if np.isnan(all_inter_list).any() or np.isinf(all_inter_list).any():
#         raise ValueError("Input data contains NaN or Inf values")
    
#     # 创建文件夹
#     output_folder = f"save/scatter_plots/{folder_name}"
#     os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在就创建

#     plt.figure(figsize=(10, 8))
    
#     added_labels = [False] * num_classes
    
#     colors = [
#         '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
#         '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#006400', '#FFC0CB'
#     ]
    
#     scaler = StandardScaler()
#     all_inter_list = scaler.fit_transform(all_inter_list)
    
#     # t-SNE 降维
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
#     all_inter1_2d = tsne.fit(all_inter_list)

#     # 绘制 all_inter_list 的散点图
#     for label in range(num_classes):
#         indices = (label_list == label)  # 获取当前标签的索引
#         plt.scatter(
#             all_inter1_2d[indices, 0],  # t-SNE 第一维
#             all_inter1_2d[indices, 1],  # t-SNE 第二维
#             label=f"Label {int(label)}",  # 确保标签是整数
#             alpha=0.5,
#             s=10,  # 点的大小
#             color=colors[int(label) % len(colors)]  # 确保索引是整数
#         )
#         added_labels[label] = True

#     # 按标签排序
#     sorted_labels = sorted(epoch_prototype.keys())

#     # 绘制 epoch_prototype 的点并按照标签顺序输出图例
#     for label in sorted_labels:
#         prototype = epoch_prototype[label].cpu().detach().numpy()
#         # print(f"Prototype shape: {prototype.shape},{prototype.ndim}")
#         if prototype.ndim == 1:
#             prototype = prototype.reshape(1, -1)  # 将 1D 数组转换为 2D 数组
#         elif prototype.ndim > 2:
#             prototype = prototype.flatten().reshape(1, -1)
#         # print(f"Prototype shape: {prototype.shape}")
        
#         # 确保 prototype 的形状与 all_inter_list 一致
#         if prototype.shape[1] != all_inter_list.shape[1]:
#             raise ValueError(f"Prototype for label {label} has {prototype.shape[1]} features, but expected {all_inter_list.shape[1]} features.")
        
#         prototype = scaler.transform(prototype)
        
#         # 如果 n_samples 为 1，直接使用 prototype 的坐标进行绘制
#         if prototype.shape[0] == 1:
#             prototype_2d = tsne.fit(np.vstack([all_inter_list, prototype]))[-1, :].reshape(1, -1)
#         else:
#             # 检查样本数量并调整 perplexity
#             n_samples = prototype.shape[0]
#             print(f"n_samples for label {label}: {n_samples}")
#             perplexity = max(1, min(30, n_samples - 1))
#             tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_jobs=-1)
#             prototype_2d = tsne.fit(prototype)
        
#         plt.scatter(
#             prototype_2d[:, 0],  # t-SNE 第一维
#             prototype_2d[:, 1],  # t-SNE 第二维
#             label=f"Prototype {int(label)}",  # 确保标签是整数
#             alpha=1.0,
#             s=50,  # 方块的大小
#             color=colors[int(label) % len(colors)],  # 确保索引是整数
#             marker='s'  # 方块标记
#         )

#     # 添加图例和标题
#     plt.legend(title="Labels", loc='best')
#     plt.title("t-SNE visualization of embeddings and prototypes")
#     plt.xlabel("")
#     plt.ylabel("")
#     plt.grid(True)

#     # 保存图像
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(os.path.join(output_folder, f"scatter_plot_{args.datasets}_{t}_{args.model}_{args.global_epoch}_{timestamp}.png"))
#     plt.close()