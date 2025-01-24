import torch
from lib.options import args_parser
from torch.utils.data import DataLoader
import torch.nn as nn
from lib.data_example import *
import random
from torch.utils.data import Subset, ConcatDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def filter_by_class(dataset, classes):
    indices = [i for i, (img, label) in enumerate(dataset) if label in classes]
    return torch.utils.data.Subset(dataset, indices)

def update_combined_fisher_information(combined_fisher_information, new_fisher_information, weight=1.0):
    # 如果combined_fisher_information是空的，初始化它为新的费舍尔信息
    combine_client_fisher_informations = []
    for i in range(len(combined_fisher_information)):
        if combined_fisher_information[i] is None:
            combined_fisher_information[i] = {}
            for name in new_fisher_information[i]:
                combined_fisher_information[i][name] = new_fisher_information[i][name] * weight
        else:
            # 将新任务的费舍尔信息累加到已有的费舍尔信息中
            for name in new_fisher_information[i]:
                combined_fisher_information[i][name] += new_fisher_information[i][name] * weight
        combine_client_fisher_informations.append(combined_fisher_information[i])

    return combine_client_fisher_informations

def combine_fisher_information(fisher_information_list, weights=None):
    combined_fisher_information = {}
    
    # 如果没有指定权重，则使用均等权重
    if weights is None:
        weights = [1.0 / len(fisher_information_list)] * len(fisher_information_list)
    
    # 初始化组合费舍尔信息矩阵
    for name in fisher_information_list[0]:
        combined_fisher_information[name] = torch.zeros_like(fisher_information_list[0][name])
    
    # 结合每个任务的费舍尔信息矩阵
    for fisher_information, weight in zip(fisher_information_list, weights):
        for name in fisher_information:
            combined_fisher_information[name] += weight * fisher_information[name]
    
    return combined_fisher_information


def multi_task_ewc_loss(model, old_models, combined_fisher_information, lambda_):
    ewc_penalty = 0
    for (name, param), (_, old_param) in zip(model.named_parameters(), old_models.named_parameters()):
        if name in combined_fisher_information:
            fisher_value = combined_fisher_information[name]
            # 计算 EWC 正则化项，确保参数变化不大
            ewc_penalty += (fisher_value * (param - old_param).pow(2)).sum()
    
    # 总损失 = EWC 正则项
    total_loss =  (lambda_ / 2) * ewc_penalty
    # print("total_loss_aaa",total_loss)
    return total_loss

def calculate_fisher_information(args, models, num_clients, server_model, train_dataset_list):
    fisher_informations = []
    data = Data(args)
    criterion = nn.CrossEntropyLoss()

    # 初始化费舍尔信息矩阵
    for j in range(num_clients):
        fisher_information = {name: torch.zeros_like(param).to(device) for name, param in models[j].named_parameters()}
        fisher_informations.append(fisher_information)
        
    
    # 将模型设置为训练模式以计算梯度
    for j in range(num_clients):
        models[j].train()
    server_model.train()
    clients_local_train_datasets=[]
    # 数据处理
    
    train_dataset_split,train_label,_,_ = data.split_data(train_dataset_list,num_clients)
    clients_local_test_datasets = []
    for i in range(num_clients):
        client_local_train_dataset = DataLoader(train_dataset_split[i], batch_size=args.batch_size,shuffle=False)
        client_local_train_dataset = iter(client_local_train_dataset)
        clients_local_train_datasets.append(client_local_train_dataset)
    server_train_label = DataLoader(train_label, batch_size=args.batch_size,shuffle=False)
        

    # 计算费舍尔信息
    for i,test_label in enumerate(server_train_label):
        test_label = test_label.to(device)
        # a_copy = []
        a = [None] * num_clients
        a_copy = [None] * num_clients

        for j in range(num_clients):
            img = next(clients_local_train_datasets[j])
            img = img.to(device)
            a[j] = models[j](img)
            a_copy[j] = a[j].clone().detach().requires_grad_()  

        # 将客户端的中间特征进行组合
        if args.baseline_type == "max":
            all_inter, _ = torch.max(torch.stack(a_copy), dim=0)
        elif args.baseline_type == "con":
            all_inter = torch.cat(a_copy, dim=1)
        else:
            all_inter1 = torch.sum(torch.stack(a_copy), dim=0)
            all_inter = all_inter1 / len(a_copy)

        # 计算服务器模型输出和损失
        out_result = server_model(all_inter)
        loss = criterion(out_result, test_label.long())

        # 清除梯度
        server_model.zero_grad()
        for j in range(num_clients):
            models[j].zero_grad()

        # 反向传播计算梯度
        loss.backward()

        # 更新费舍尔信息矩阵
        for j in range(num_clients):
            if a_copy[j].grad is not None:
                g = a_copy[j].grad
            else:
                print(f"Warning: Gradient for a_copy[{j}] is None")
                a_copy[j] = torch.zeros_like(a_copy[j], dtype=torch.float64, requires_grad=True)
                g = a_copy[j].grad
            
            a[j].backward(g)

            # 累加每个参数的梯度平方值
            for name, param in models[j].named_parameters():
                if param.grad is not None:
                    fisher_informations[j][name] += param.grad.pow(2)
                    # print("fisher_informations[j][name]",fisher_informations[j][name])

    # 计算费舍尔信息的平均值
    num_samples = len(server_train_label)
    for j in range(num_clients):
        fisher_informations[j] = {name: fisher / num_samples for name, fisher in fisher_informations[j].items()}
        # print("fisher_informations_aaa",fisher_informations[j])

    return fisher_informations

def combine_client_fisher_information(task_fisher_information, weights=None):
    num_tasks = len(task_fisher_information)
    num_clients = len(task_fisher_information[0])
    
    # 如果没有指定权重，则使用均等权重
    if weights is None:
        # weights = [1.0 / num_tasks] * num_tasks
        weights = [1.0] * num_tasks

    # 初始化每个客户端的组合费舍尔信息
    combined_client_fisher_information = []
    for client_index in range(num_clients):
        combined_fisher_information = {}
        
        # 初始化每个参数的费舍尔信息为0
        for name in task_fisher_information[0][client_index]:
            combined_fisher_information[name] = torch.zeros_like(task_fisher_information[0][client_index][name])
        
        # 结合每个任务的费舍尔信息
        for task_index in range(num_tasks):
            current_fisher_information = task_fisher_information[task_index][client_index]
            weight = weights[task_index]

            for name in current_fisher_information:
                combined_fisher_information[name] += weight * current_fisher_information[name]
        
        # 添加到组合费舍尔信息列表中
        combined_client_fisher_information.append(combined_fisher_information)
    
    return combined_client_fisher_information
                    
                    
                