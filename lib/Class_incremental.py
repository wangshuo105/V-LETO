# This is a sample Python script.
import pandas as pd
import torch
from lib.data_example import *
from lib.options import args_parser
from random import choice
from lib.models.models import *
from lib.models.models_cifar import *
# from lib.models.resnet_cifar import *
from lib.models.resnet_mnist import *
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
from lib.Tool import *
from lib.LocalUpdate import *
import gc




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def save_img(image):
    img = image
    if image.shape[0] == 1:
        pixel_min = torch.min(img)
        img -= pixel_min
        pixel_max = torch.max(img)
        img /= pixel_max
        io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img.squeeze().numpy()))
    else:
        img = image.numpy()
        img = img.transpose(1, 2, 0)
        pixel_min = np.min(img)
        img -= pixel_min
        pixel_max = np.max(img)
        img /= pixel_max
        io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img))



def transpose_2d(data):
    transposed = []
    for i in range(len(data[0])):
        new_row = []
        for row in data:
            new_row.append(row[i])
        transposed.append(new_row)
    return transposed




def save_model(model, t, party):
    # 确保保存目录存在
    save_dir = f"save_model_class/{t}/{args.datasets}"
    os.makedirs(save_dir, exist_ok=True)
    # 自定义保存文件名，包含 epoch 和其他信息
    filename = os.path.join(save_dir, f"{party}_model_{args.model}_{args.datasets}_{args.lambda_proto_aug}_{args.k_0}_{args.alpha}_{args.lr}_task_{t}.pth")
    
    # 保存模型的 state_dict
    torch.save(model.state_dict(), filename)
    del model  # 删除模型对象
    gc.collect()  # 调用垃圾回收
    # print(f"Model saved to {filename}")


def load_model(model, model_path):
    # 加载保存的 state_dict
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    # model.eval()  # 切换到评估模式
    return model

def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        print(f"f folder {folder_name} exist")
        
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


def old_prototype_current_compute(global_prototype, keys_in_global_not_in_current, euclidean_distance):
    old_prototype_current = {}
    for key in keys_in_global_not_in_current:
        if key in global_prototype:
            # 计算新的值：global_prototype[key] + A * euclidean_distance[key]
            new_value = global_prototype[key] + args.r * euclidean_distance
            # 将键和值添加到 old_prototype_current 中
            old_prototype_current[key] = new_value
    return old_prototype_current

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

def server_model_create(args):
    resnet18_top = resnet18()[1]
    resnet50_top = resnet50()[1]
    resnet101_top = resnet101()[1]
    if args.model =="mlp": 
        server_model = mlp_top()
    elif args.model == "cnn":
        server_model = cnn_top()
    elif args.model == "lenet":
        server_model = lenet_top()
    elif args.model == "resnet18":
        server_model = resnet18_top
    elif args.model == "resnet50":
        server_model = resnet50_top
    elif args.model == "resnet101":
        server_model = resnet101_top
    else:
        print("the model not existing")
    return server_model

def all_inter_aggerate(a,args):
    if args.baseline_type=="max":
        all_inter,_ = torch.max(torch.stack(a),dim=0)
    elif args.baseline_type=="con":
        all_inter = torch.cat(a, dim=1)
    else:
        all_inter1 = torch.sum(torch.stack(a), dim=0)
        all_inter = all_inter1 / len(a)
    return all_inter

def select_important_parameters(fisher_information, threshold=0.01):
    important_params = {}
    for name, fisher_value in fisher_information.items():
        # 选择费舍尔信息值大于阈值的参数
        important_params[name] = (fisher_value > threshold).float()
    return important_params

def single_bottom_create(num_clients):
    resnet18_bottom = resnet18()
    resnet50_bottom = resnet50()
    resnet101_bottom = resnet101()
    if args.model =="mlp":
        if args.datasets == "cifar10":
            dim_in = int(32/num_clients)
            model = mlp_cifar_bottom(dim_in)
        else:
            dim_in = int(28/num_clients)
            model = mlp_bottom(dim_in)
    elif args.model == "cnn":
        if args.datasets == "cifar10":
            model = cnn_cifar_bottom()
        else:
            model = cnn_bottom()
    elif args.model == "lenet":
        if args.datasets == "cifar10":
            model = lenet_cifar_bottom()
        else:
            model = lenet_bottom()
    elif args.model == "resnet18":
        model = resnet18_bottom
    elif args.model == "resnet50":
        model = resnet50_bottom
    elif args.model == "resnet101":
        model = resnet101_bottom
    else:
        print("the model not existing")
    model.to(device)
    return model

def bottom_model_create(num_clients,args):
    models = []
    optimizers = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18_bottom = resnet18()[0]
    resnet50_bottom = resnet50()[0]
    resnet101_bottom = resnet101()[0]
    three_image =["cifar10","cifar100","cinic10"]
    if args.train_type == "heter":
        if args.model == "mlp" or args.model == 'cnn' or args.model == 'lenet':
            if args.datasets in three_image:
                model_list = [mlp_cifar_bottom(),lenet_cifar_bottom(),cnn_cifar_bottom()]
            else:
                dim_in = int(28/num_clients)
                print("dim_in::",dim_in)
                model_list = [mlp_bottom(dim_in),lenet_bottom(),cnn_bottom()]
        else:
            model_list = [resnet18_bottom,resnet50_bottom,resnet101_bottom] 
    
    for i in range(num_clients):
        if args.train_type == "heter":
            model = model_list[i%3]
        else:
            if args.model =="mlp":
                if args.datasets in three_image:
                    dim_in = int(32/num_clients)
                    model = mlp_cifar_bottom(dim_in)
                else:
                    dim_in = int(28/num_clients)
                    model = mlp_bottom(dim_in)
            elif args.model == "cnn":
                if args.datasets in three_image:
                    model = cnn_cifar_bottom()
                else:
                    model = cnn_bottom()
            elif args.model == "lenet":
                if args.datasets in three_image:
                    model = lenet_cifar_bottom()
                else:
                    model = lenet_bottom()
            elif args.model == "resnet18":
                model = resnet18_bottom
            elif args.model == "resnet50":
                model = resnet50_bottom
            elif args.model == "resnet101":
                model = resnet101_bottom
            else:
                print("the model not existing")
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        for param in model.parameters():
            if param.is_cuda:
                print("Model is on GPU")
                break
            else:
                print("Model is on CPU")
        models.append(model)
        optimizers.append(optimizer)
    return models,optimizers


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

def prototype_compute(all_inter,label):
    unique_labels = torch.unique(label)
    prototypes = {}
    for label_val in unique_labels:
        # 获取属于当前标签的样本索引
        idx = (label == label_val)
        
        # 取出所有属于该类别的样本
        samples_in_class = all_inter[idx]
        
        # 计算该类别的原型（均值）
        prototypes[label_val.item()] = samples_in_class.mean(dim=0)
    # print("prototype[0]",prototypes[0])
    return prototypes

def Class_train(args, logger):
    data = Data(args)
    # folder_name1 = args.folder_name
    num_clients = args.num_user
    
    train_dataset, test_dataset = data.load_dataset(args)

    if args.datasets == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
        
    train_dataset_list = []
    if args.class_type == 'class_task':
        for i in range(args.task):
            filtered_train = filter_by_class(train_dataset, [i*2,i*2+1,i*2+2,i*2+3])
            train_indices = list(filtered_train.indices)
            random.shuffle(train_indices)
            filtered_train = Subset(filtered_train.dataset, train_indices)
            train_dataset_list.append(filtered_train)
    else:
        train_dataset_list.append(train_dataset)
    print("len(train_dataset_list)",len(train_dataset_list))    
        
    test_dataset_list = []
    if args.class_type == 'class_task':
        for j in range(args.task):
            logger.info(f'test_aaa:{j}')
            filtered_test = filter_by_class(test_dataset, [j*2,j*2+1,j*2+2,j*2+3])
            test_indices = list(filtered_test.indices)
            random.shuffle(test_indices)
            filtered_test = Subset(filtered_test.dataset, test_indices)
            test_dataset_list.append(filtered_test)
    else:
        test_dataset_list.append(test_dataset)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    
    models, optimizers = bottom_model_create(num_clients,args)

    server_model = server_model_create(args)

    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()
    

    init_acc = 0
    
    global_prototype={}
    global current_prototype,number_data
    old_models = []
    task_fisher_information = []
    k_0 = args.k_0  
    alpha = args.alpha 
    
    
    folder_name = f'save/VFCL_prototype/Class/{args.datasets}'
    create_folder(folder_name)
    

    excel_file_path = f'./save/VFCL_prototype/Class/{args.datasets}/VFCL_prototype_{args.datasets}_{args.model}_{args.global_epoch}_{args.weight_type}_task_{args.task}_{args.task_type}_{args.num_user}_{args.k_0}_{args.alpha}_train_{time.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx'
    train_row = 0
    test_row = 0
  
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    
        for t in range(args.task):
            train_dataset_split,train_label,_,_ = data.split_data(train_dataset_list[t],num_clients)
            old_models = []
            
            if t !=0:
                for j in range(num_clients):
                    save_dir = f"save_model_class/{t-1}/{args.datasets}"
                    filename = os.path.join(save_dir, f"{j}_model_{args.model}_{args.datasets}_{args.lambda_proto_aug}_{args.k_0}_{args.alpha}_{args.lr}_task_{t-1}.pth")
                    models[j] = load_model(models[j],filename)
                    old_models.append(copy.deepcopy(models[j]))
                save_dir = f"save_model_class/{t-1}/{args.datasets}"
                filename = os.path.join(save_dir, f"active_model_{args.model}_{args.datasets}_{args.lambda_proto_aug}_{args.k_0}_{args.alpha}_{args.lr}_task_{t-1}.pth")
                server_model = load_model(server_model, filename)
                old_server_model = copy.deepcopy(server_model)
                old_server_model.eval()
                logger.info("load_model success")
            
            
            for j in range(num_clients):
                 old_models.append(copy.deepcopy(models[j]))
            old_server_model = copy.deepcopy(server_model)
            old_server_model.eval()
            if t == 0:
                epoch_prototype = []    
            
            for epoch in range(args.global_epoch):
                batch_loss = []
                batch_acc = []
                clients_local_train_datasets=[]
                
                embedding_list = []
                label_list = []
                for i in range(num_clients):
                    client_local_train_dataset = DataLoader(train_dataset_split[i], batch_size=args.batch_size,shuffle=False)
                    client_local_train_dataset = iter(client_local_train_dataset)
                    clients_local_train_datasets.append(client_local_train_dataset)
                server_train_label = DataLoader(train_label, batch_size=args.batch_size,shuffle=False)
                
                
                clients_local_test_datasets=[]
                test_dataset_split,test_label,_,_ = data.split_data(test_dataset_list[t],num_clients)
                for j in range(num_clients):
                    client_local_test_dataset = DataLoader(test_dataset_split[j], batch_size=10*args.batch_size,shuffle=False)
                    client_local_test_dataset = iter(client_local_test_dataset)
                    clients_local_test_datasets.append(client_local_test_dataset)
                
               
                  
                    
                
                for i,label in enumerate(server_train_label):
                    label = label.to(device)
                    agg_protos_label = {}
                    
                    for j in range(num_clients):
                        optimizers[j].zero_grad()
                        models[j].train()
                    server_optimizer.zero_grad()
                    server_model.train()
                    a = [None] * num_clients
                    a_copy = [None] * num_clients
                    old_a = [None] * num_clients
                    loss_al = [None] * num_clients
                    prototype_loss_value = 0
                    
                    for j in range(num_clients):
                        img = next(clients_local_train_datasets[j])
                        number_data = img.shape[0]
                        img = img.to(device)
                        a[j] = models[j](img)
                        a_copy[j] = a[j].clone().detach().requires_grad_()  
                        if t !=0: 
                            old_a[j] = old_models[j](img)
                
                    all_inter = all_inter_aggerate(a_copy,args)
                    
                    
                    current_prototype = prototype_compute(all_inter,label)
            
                    
                    
                    
                
                    
                    contrastive_loss = ContrastiveLoss(margin=1.0)
                    device = all_inter.device
                    if len(epoch_prototype) == 0:
                        prototype_loss_value = 0
                    else:
                        proto_new = all_inter.clone().to(device)
                        j1 = 0
                        for l in label:
                            if l.item() in epoch_prototype.keys():
                                proto_new[j1, :] = epoch_prototype[l.item()].to(device)
                            j1 += 1

        
                        labels = torch.zeros(all_inter.size(0), device=device)
                        for il, l in enumerate(label):
                            if l.item() in epoch_prototype.keys():
                                labels[il] = 1

                        prototype_loss_value = contrastive_loss(proto_new, all_inter, labels)
                    
                
                    
                    
                    
                    if t!=0:
                        old_all_inter = all_inter_aggerate(old_a)
                    
                    if (t != 0 and args.task_type == "feature_enhance"):
                        existing = 1
                    elif (t != 0 and args.task_type == "class_enhance"):
                        pre_prototype = prototype_compute(old_all_inter,label)  
                        current_prototype = prototype_compute(all_inter,label)
                        euclidean_distance = distance_prototype(pre_prototype,current_prototype)
                        keys_in_global_not_in_current = global_prototype.keys() - current_prototype.keys()
                        old_prototype_current = old_prototype_current_compute(global_prototype,keys_in_global_not_in_current,euclidean_distance)
                        # old_prototype_current = old_prototype_current_compute_PAS(global_prototype,keys_in_global_not_in_current,r)
                        proto_aug, proto_aug_label = proto_aug_compute(old_prototype_current,t,number_data)
                        proto_aug = proto_aug.to(device)
                        proto_aug_label = proto_aug_label.to(device)
                        
                    # if epoch == args.global_epoch - 1:
                    #     if i == 0:
                    #         all_inter_save = all_inter.cpu()
                    #         label_save = label.cpu()
                    #         embedding_list.append(all_inter_save)
                    #         label_list.append(label_save)
                    #         if t!= 0:
                    #             proto_aug_save = proto_aug.cpu()
                    #             proto_aug_label_save = proto_aug_label.cpu()
                    #             embedding_list.append(proto_aug_save)
                    #             label_list.append(proto_aug_label_save)
                    #     else:
                    #         if len(embedding_list) == 0:
                    #             print("embedding_all_list is empty",i)
                    #         embedding_all_list = torch.cat(embedding_list, dim=0) 
                    #         label_all_list = torch.cat(label_list, dim=0)
                    #         unique_labels = np.unique(label_all_list)
                    #         for l in unique_labels:
                    #             indices = np.where(label_all_list == l)[0]
                    #             if len(indices) < 4000:
                    #                 all_inter_save = all_inter.cpu()
                    #                 label_save = label.cpu()
                    #                 embedding_list.append(all_inter_save)
                    #                 label_list.append(label_save)
                    #                 if t!=0:
                    #                     proto_aug_save = proto_aug.cpu()
                    #                     proto_aug_label_save = proto_aug_label.cpu()
                    #                     embedding_list.append(proto_aug_save)
                    #                     label_list.append(proto_aug_label_save)
                    #             print("len(indices)",len(indices))
                    #             break
                    #         for l in unique_labels:
                    #             print(unique_labels)
                    #             indices = np.where(label_all_list == l)[0]
                    #             print("each_len(indices)",len(indices))
        
                        
                    out_result = server_model(all_inter)
                    
                    loss_a = criterion(out_result, label.long())
                    
                    if (t != 0 and (args.task_type == "feature_enhance" or args.task_type =="class_enhance")):
                        proto_aug_result = server_model(proto_aug)
                        proto_aug_loss = criterion(proto_aug_result, proto_aug_label)
                       
                        loss =  loss_a + args.lambda_proto_aug*proto_aug_loss
                    else:
                        loss = loss_a +  prototype_loss_value
                        
                    if (t != 0 and (args.task_type == "feature_enhance" or args.task_type =="class_enhance")):
                        _, predicted_aug = torch.max(proto_aug_result.data, 1)
                        _, predicted_a = torch.max(out_result.data, 1)
                        correct = (predicted_a == label).sum().item() + (predicted_aug == proto_aug_label).sum().item()
                        total = label.size(0) + proto_aug_label.size(0)

                    else:
                        _, predicted_a = torch.max(out_result.data, 1)
                        correct = (predicted_a == label).sum().item()
                        total = label.size(0)
                    
                    
                    batch_loss.append(loss.item())
                    batch_acc.append(100 * correct/ total)
                    
                    loss.backward(retain_graph=True)
                    
                    
                    
                    for j in range(num_clients):
                        if a_copy[j].grad is not None:
                            g = a_copy[j].grad
                        else:
                            logger.info(f"Warning: Gradient for a_copy[{j}] is None")
                            a_copy[j] = torch.zeros_like(a_copy[j], dtype=torch.float64, requires_grad=True)
                            g = a_copy[j].grad
                        
                        a[j].backward(g)
                        
                        if t != 0:
                            fisher_values = torch.cat([info.view(-1) for info in combined_fisher_information[j].values()])
                            mean_fisher = fisher_values.mean().item()
                            std_fisher = fisher_values.std().item()
                            k_log = k_0 + alpha * math.log(t + 1)
                            threshold = mean_fisher - k_log * std_fisher
                            threshold = max(threshold, 0) 
                            important_params = select_important_parameters(combined_fisher_information[j], threshold)
                            
                            with torch.no_grad():
                                for name, param in models[j].named_parameters():
                                    if name in important_params:
                                        param.grad = (1 - important_params[name])

                        
                    server_optimizer.step()
                    for j in range(num_clients):
                        optimizers[j].step()
                        
                    for i in range(len(label)):
                        if label[i].item() in agg_protos_label:
                            agg_protos_label[label[i].item()].append(all_inter[i,:])
                        else:
                            agg_protos_label[label[i].item()] = [all_inter[i,:]]
                
                epoch_prototype = agg_func(agg_protos_label) 
                    
                    
                init_acc = sum(batch_acc)/(len(batch_acc))
                init_loss= sum(batch_loss)/len(batch_loss)
                
                
                
                df_train = pd.DataFrame({
                    'Task' : [t],
                    'Epoch': [epoch+1], 
                    'Training Accuracy': [init_acc], 
                    'Training Loss': [init_loss]
                })
                if epoch == 0 and t == 0:
                    df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=0, header=True)
                else:
                    df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=train_row, header=False)
                train_row += 1
                
                logger.info(f'Training Task: {t},Epoch:{epoch+1},Training Accuracy:{init_acc},Training Loss:{init_loss}')
                
                test_task_acc, test_task_loss = test_model(server_model, num_clients, models, test_dataset, args, t, logger, epoch,data)
                

                for tt in range(len(test_task_acc)):
                    df_test = pd.DataFrame({
                        'Training Task':[t],
                        'Epoch': [epoch+1], 
                        'Testing Task':[tt],
                        'Test Accuracy': [test_task_acc[tt]], 
                        'Test Loss': [test_task_loss[tt]]
                    })
                    if epoch == 0 and tt == 0 and t == 0:
                        df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=0, header=True)
                        df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=1, header=False)
                        test_row += 2
                    else:
                        df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=test_row, header=False)
                        test_row += 1
                        
                    logger.info(f'[Training Task: {t}, Epoch:{epoch+1}, Testing Task:{tt},Test Accuracy:{test_task_acc[tt]},Test Loss:{test_task_loss[tt]}]')


                for key, value in current_prototype.items():
                    if key in global_prototype:
                        global_prototype[key] = global_prototype[key]
                    else:
                        global_prototype[key] = value

                        
               
                
            
            torch.cuda.empty_cache()
            
            fisher_informations = calculate_fisher_information(args, models, num_clients, server_model, test_dataset_list[t])
            if t == 0:
                old_fisher_informations = calculate_fisher_information(args, models, num_clients, server_model, test_dataset_list[t])
                combined_fisher_information = old_fisher_informations
            else:
                new_fisher_informations=calculate_fisher_information(args, models, num_clients, server_model, test_dataset_list[t])
                combined_fisher_information = update_combined_fisher_information(combined_fisher_information, new_fisher_informations)
            
            task_fisher_information.append(fisher_informations)

            
                
            for j in range(num_clients):
                save_model(models[j],t,j)
                print("save_model success",len(models))
            # if len(models) == num_clients:
            #     for j in range(num_clients):
            #         del models[j]  # 释放模型的内存
            # for j in range(num_clients):
            #     del models[j]  # 释放模型的内存
            save_model(server_model,t,"active")
            
        
                          



def test_model(server_model, num_clients, models, test_dataset, args, t, logger, ep,data):
    test_acc = []
    test_loss = []
    correct = 0
    loss = 0
    total = 0
    test_task_loss = []
    test_task_acc = []

    server_model.eval()
    for j in range(num_clients):
        models[j].eval()
    
    test_dataset_list = []
    current_labels = []
    for j in range(t+1):
        print("test_aaa", j)
        current_label = [j*2,j*2+1,j*2+2,j*2+3]
        print("current_label", current_label)
        current_labels = list(set(current_labels) | set(current_label))
        filtered_test = filter_by_class(test_dataset, current_label)
        test_indices = list(filtered_test.indices)
        random.shuffle(test_indices)
        filtered_test = Subset(filtered_test.dataset, test_indices)
        test_dataset_list.append(filtered_test)
        
    print(current_labels)
    all_task_test_dataset = filter_by_class(test_dataset, current_labels)
    test_indices = list(all_task_test_dataset.indices)
    random.shuffle(test_indices)
    all_task_test_dataset = Subset(filtered_test.dataset, test_indices)
    test_dataset_list.append(all_task_test_dataset)
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for t_test in range(t+2):
            clients_local_test_datasets=[]
            test_dataset_split,test_label,_,_ = data.split_data(test_dataset_list[t_test],num_clients)
            for j in range(num_clients):
                client_local_test_dataset = DataLoader(test_dataset_split[j], batch_size=10*args.batch_size,shuffle=False)
                client_local_test_dataset = iter(client_local_test_dataset)
                clients_local_test_datasets.append(client_local_test_dataset)
            server_test_lable = DataLoader(test_label, batch_size=10*args.batch_size,shuffle=False)
            centers = [None] * num_clients
            init_acc = 0
            for i, test_label in enumerate(server_test_lable): 
                test_label = test_label.to(device)
                a = []
                a_copy = []
                for j in range(num_clients):
                    img = next(clients_local_test_datasets[j])
                    img = img.to(device)
                    # inter = models[j](img)
                    inter = models[j](img)
                    a_copy.append(inter.clone().detach().requires_grad_())
                
                #替换测试数据为原型
                all_inter = all_inter_aggerate(a_copy,args)
                out_result = server_model(all_inter)
                loss = criterion(out_result, test_label.long())
                _, predicted_a = torch.max(out_result.data, 1)
                correct = (predicted_a == test_label).sum().item()
                total = test_label.size(0)
                
                # logger.info('Test_Task:{:d},Test:[{:d}], loss: {:.3f}, accure : {:.3f}'.format(t_test, i, loss.item(), 100 * correct /total))
                test_loss.append(loss.item())
                init_acc = correct / total
                test_acc.append(100 * correct /total)
            avg_test_loss = sum(test_loss) / len(test_loss)
            avg_test_acc = sum(test_acc) / len(test_acc)
            # logger.info(f'[Testing Epoch: {ep+1}], loss: {avg_test_loss}, accuracy: {avg_test_acc}')
            test_task_acc.append(avg_test_acc)
            test_task_loss.append(avg_test_loss)
    return test_task_acc, test_task_loss

if __name__ == '__main__':
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.cuda.manual_seed(3407)
    
    args = args_parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = args_parser()
    logger = setup_logger(args)
    logger.info(device)
    logger.info("命令行参数:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    # PraVFed_train(args,logger)
    Class_train(args,logger)
    
