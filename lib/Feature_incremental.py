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
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset
# from lib.prototype import *
from lib.Tool import *




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        print(f"f folder {folder_name} exist")




def transpose_2d(data):
    transposed = []
    for i in range(len(data[0])):
        new_row = []
        for row in data:
            new_row.append(row[i])
        transposed.append(new_row)
    return transposed



def save_model(model, t, party):
    save_dir = f"save_model/feature/{t}/{args.datasets}"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{party}_model_task_{t}.pth")
    
    torch.save(model.state_dict(), filename)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

        
def filter_by_class(dataset, classes):
    indices = [i for i, (img, label) in enumerate(dataset) if label in classes]
    return torch.utils.data.Subset(dataset, indices)


def old_prototype_current_compute(global_prototype, keys_in_global_not_in_current, euclidean_distance):
    old_prototype_current = {}
    for key in keys_in_global_not_in_current:
        if key in global_prototype:
            new_value = global_prototype[key] + args.r * euclidean_distance
            old_prototype_current[key] = new_value
    return old_prototype_current

def all_prototype_current_compute(global_prototype, union_prototype, euclidean_distance):
    old_prototype_current = {}
    for key in union_prototype:
        if key in global_prototype:
            new_value = global_prototype[key] + args.r * euclidean_distance
            old_prototype_current[key] = new_value
        else:
            old_prototype_current[key] = union_prototype[key]
    return old_prototype_current

def server_model_create():
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


def all_inter_aggerate(a):
    if args.baseline_type=="max":
        all_inter,_ = torch.max(torch.stack(a),dim=0)
    elif args.baseline_type=="con":
        all_inter = torch.cat(a, dim=1)
    else:
        all_inter1 = torch.sum(torch.stack(a), dim=0)
        all_inter = all_inter1 / len(a)
    return all_inter1


def select_important_parameters(fisher_information, threshold=0.01):
    important_params = {}
    for name, fisher_value in fisher_information.items():
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


criterion = nn.CrossEntropyLoss()


def distribution_regularizer(output, target_distribution):
    output_distribution = torch.mean(output, dim=0)
    loss = F.mse_loss(output_distribution, target_distribution)
    return loss

def obtain_target_distribution(device, label_num, embedding_dim):
    target_distributions = {}
    for label in range(label_num):
        target_distributions[label] = torch.normal(mean=label, std=1, size=(embedding_dim,)).to(device)
    return target_distributions
def bottom_model_create(num_clients):
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

def prototype_compute(all_inter,label):
    unique_labels = torch.unique(label)
    prototypes = {}
    for label_val in unique_labels:
        idx = (label == label_val)
        samples_in_class = all_inter[idx]
        prototypes[label_val.item()] = samples_in_class.mean(dim=0)
    return prototypes


def Feature_train(args,logger):
    data = Data(args)
    num_clients = args.num_user
    train_dataset, test_dataset = data.load_dataset(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    
    models, optimizers = bottom_model_create(num_clients)
    
    folder_name = f'save/VFCL_Two/Feature/{args.datasets}'
    create_folder(folder_name)
    
    excel_file_path = f'./save/VFCL_Two/Feature/{args.datasets}/VFCL_{args.datasets}_{args.model}_{args.global_epoch}_{args.weight_type}_task_{args.task}_{args.task_type}_{args.num_user}_{args.k_0}_{args.alpha}_train_{time.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx'
    train_row = 0
    test_row = 0

    server_model = server_model_create()
    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr)
    
    
    init_acc = 0
    
    global_prototype={}
    global current_prototype,number_data
    
    client_global_prototype = []
    client_global_prototype_error=[]
    
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        for t in range(args.task):
            task_client=[]
            task_client=[t]
            epoch_prototype = []
            
            for epoch in range(args.global_epoch):
                batch_loss = []
                batch_acc = []
                clients_local_train_datasets=[]
                train_dataset_split,train_label,_,_ = data.split_data(train_dataset,num_clients)    
                embedding_list = []
                
                for i in range(num_clients):
                    client_local_train_dataset = DataLoader(train_dataset_split[i], batch_size=args.batch_size,shuffle=False)
                    client_local_train_dataset = iter(client_local_train_dataset)
                    clients_local_train_datasets.append(client_local_train_dataset)
                server_train_label = DataLoader(train_label, batch_size=args.batch_size,shuffle=False)
                
                embedding_list = []
                label_list = []
                embedding_prototype_list = []
                embedding_prototype_two_list =[]
                for i,train_label in enumerate(server_train_label):
                    agg_protos_label = {}
                    train_label = train_label.to(device)
                    
                    for j in task_client:
                        optimizers[j].zero_grad()
                        models[j].train()
                    server_optimizer.zero_grad()
                    server_model.train()
                    a = [None] * num_clients
                    a_copy = [None] * num_clients
                    
                    for j in task_client:
                        img = next(clients_local_train_datasets[j])
                        number_data = img.shape[0]
                        img = img.to(device)
                        a[j] = models[j](img)
                        a_copy[j] = a[j].clone().detach().requires_grad_()  
                        all_inter1 = a_copy[j]
                    
                    current_prototype = prototype_compute(all_inter1, train_label)
                    
                    
                    
                    if epoch == args.global_epoch - 1:
                        embedding_list.append(all_inter1)
                        label_list.append(train_label)
                    
                
                    
                    
                    if (t != 0):
                        for k in client_global_prototype:
                            prototype_matrix_client = torch.stack([k[label.item()] for label in train_label])
                            if k is client_global_prototype[0]:
                                prototype_matrix = prototype_matrix_client
                            else:
                                prototype_matrix = prototype_matrix + prototype_matrix_client
                            

                        if (args.Abaltion_LCE == 0 and args.Abaltion_LF==1 and args.Abaltion_LMO==1):
                            all_inter = prototype_matrix
                        elif (args.Abaltion_LCE == 1 and args.Abaltion_LF==0 and args.Abaltion_LMO==1):
                            all_inter = all_inter1
                        else:
                            all_inter = 0.5 * all_inter1 + 0.5 * prototype_matrix
                    else:
                        all_inter = all_inter1     
                    
                    if epoch == args.global_epoch - 1 and t !=0 :
                        embedding_prototype_list.append(prototype_matrix)
                        embedding_prototype_two_list.append(all_inter)

                        
                    current_prototype = prototype_compute(all_inter, train_label)
                    
                    contrastive_loss = ContrastiveLoss(margin=1.0)
                    device = all_inter1.device
                    if len(epoch_prototype) == 0:
                        prototype_loss_value = 0
                    else:
                        proto_new = all_inter1.clone().to(device)
                        j1 = 0
                        for l in train_label:
                            if l.item() in epoch_prototype.keys():
                                proto_new[j1, :] = epoch_prototype[l.item()].to(device)
                            j1 += 1

                        labels = torch.zeros(all_inter1.size(0), device=device)
                        for i, l in enumerate(train_label):
                            if l.item() in epoch_prototype.keys():
                                labels[i] = 1

                        
                        
                
                    
                    out_result = server_model(all_inter)      
                    loss_a = criterion(out_result, train_label.long())
                    loss = loss_a 
                    _, predicted_a = torch.max(out_result.data, 1)
                    correct = (predicted_a == train_label).sum().item()
                    total = train_label.size(0)   
                    batch_loss.append(loss.item())
                    batch_acc.append(100 * correct/ total)
                    loss.backward(retain_graph=True)
                    
                    for j in task_client:
                        if a_copy[j].grad is not None:
                            g = a_copy[j].grad
                        else:
                            print(f"Warning: Gradient for a_copy[{j}] is None")
                            a_copy[j] = torch.zeros_like(a_copy[j], dtype=torch.float64, requires_grad=True)
                            g = a_copy[j].grad
                        a[j].backward(g)    
                          
                    server_optimizer.step()
                    for j in task_client:
                        optimizers[j].step()
                        
                    for i in range(len(train_label)):
                        if train_label[i].item() in agg_protos_label:
                            agg_protos_label[train_label[i].item()].append(all_inter[i,:])
                        else:
                            agg_protos_label[train_label[i].item()] = [all_inter[i,:]]
                            
                epoch_prototype, error = agg_func_error(agg_protos_label)   
                # print("error",error)
                agg_protos_label = {}
                
                init_acc = sum(batch_acc)/(len(batch_acc))
                init_loss = sum(batch_loss) / (len(batch_loss))
                
                df_train = pd.DataFrame({
                    'Task' : [t],
                    'Epoch': [epoch+1], 
                    'Training Accuracy': [init_acc], 
                    'Training Loss': [init_loss]
                })
                if epoch == 0 and t == 0:
                    df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=0, header=True)
                    df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=1, header=False)
                    train_row += 1
                else:
                    df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=train_row, header=False)
                train_row += 1
                
                logger.info(f'Training Task: {t},Epoch:{epoch+1},Training Accuracy:{init_acc},Training Loss:{init_loss}')
                

                test_task_acc, test_task_loss = test_model(server_model, num_clients, models, test_dataset, args, t, logger, task_client,data,client_global_prototype,client_global_prototype_error,current_prototype)

                # for tt in range(len(test_task_acc)):
                df_test = pd.DataFrame({
                    'Training Task':[t],
                    'Epoch': [epoch+1], 
                    'Testing Task':[t],
                    'Test Accuracy': [test_task_acc], 
                    'Test Loss': [test_task_loss]
                })
                if epoch == 0 and t == 0:
                    df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=0, header=True)
                    df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=1, header=False)
                    test_row += 1
                else:
                    df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=test_row, header=False)
                test_row += 1
                    
                logger.info(f'[Training Task: {t}, Epoch:{epoch+1}, Testing Task:{t},Test Accuracy:{test_task_acc},Test Loss:{test_task_loss}]')
            
            
            

            for key, value in current_prototype.items():
                if key in global_prototype:
                    global_prototype[key] = value
                else:
                    global_prototype[key] = value
            
            client_global_prototype.append(epoch_prototype)
            client_global_prototype_error.append(error)
            
            
                
            for j in task_client:
                save_model(models[j],t,j)
            save_model(server_model,t,"active")

            
        

def test_model(server_model, num_clients, models, test_dataset, args, t, logger, task_client,data,client_global_prototype,client_global_prototype_error,current_prototype):
    test_acc = []
    test_loss = []
    correct = 0
    loss = 0
    total = 0
    test_task_loss = []
    test_task_acc = []

    test_task_client = []
    for j in range(t+1):
        test_task_client.append(j)
        
    for j in test_task_client:
        if j not in task_client:
            save_dir = f"save_model/feature/{j}/{args.datasets}"
            filename = os.path.join(save_dir, f"{j}_model_task_{j}.pth")
            models[j] = load_model(models[j],filename)
            models[j].eval()
    
    
    for j in task_client:
        models[j].eval()
    
    server_model.eval()

    
    test_dataset_list = []
    test_dataset_list.append(test_dataset)
    
    
    with torch.no_grad():
        clients_local_test_datasets=[]
        test_dataset_split,test_label,_,_ = data.split_data(test_dataset,num_clients)
        for j in range(num_clients):
            client_local_test_dataset = DataLoader(test_dataset_split[j], batch_size=10*args.batch_size,shuffle=False)
            client_local_test_dataset = iter(client_local_test_dataset)
            clients_local_test_datasets.append(client_local_test_dataset)
        server_test_lable = DataLoader(test_label, batch_size=10*args.batch_size,shuffle=False)
        
        
        
        init_acc = 0
        for i, test_label in enumerate(server_test_lable): 
            test_label = test_label.to(device)
            a = []
            a_copy = []
            if args.Abaltion_LMO==0:
                for j in test_task_client:
                    img = next(clients_local_test_datasets[j])
                    img = img.to(device)
                    inter = models[j](img)
                    a_copy.append(inter.clone().detach().requires_grad_())    
            else:
                for j in task_client:
                    img = next(clients_local_test_datasets[j])
                    img = img.to(device)
                    inter = models[j](img)
                    a_copy.append(inter.clone().detach().requires_grad_())    
            all_inter = all_inter_aggerate(a_copy)
            obtain_label_test = find_closest_label(all_inter,current_prototype)

                
            if (t != 0 and args.Abaltion_LMO!=0):
                test_current_matrix = torch.stack([current_prototype[label.item()] for label in test_label])
                for k in client_global_prototype:
                    prototype_matrix_client = torch.stack([k[label.item()] for label in obtain_label_test])
                    noise = test_current_matrix - prototype_matrix_client
                    prototype_matrix_client = prototype_matrix_client + noise  
                    if k is client_global_prototype[0]:
                        prototype_matrix = prototype_matrix_client
                    else:
                        prototype_matrix = prototype_matrix + prototype_matrix_client
                
              
              
                if (args.Abaltion_LCE == 0 and args.Abaltion_LF==1 and args.Abaltion_LMO==1):
                    all_inter = prototype_matrix
                elif (args.Abaltion_LCE == 1 and args.Abaltion_LF==0 and args.Abaltion_LMO==1):
                    all_inter = all_inter
                else:
                    all_inter = (1-args.lambda_proto_aug) * all_inter + args.lambda_proto_aug * prototype_matrix
            else:
                all_inter = all_inter
            
            out_result = server_model(all_inter)
            loss = criterion(out_result, test_label.long())
            _, predicted_a = torch.max(out_result.data, 1)
            correct = (predicted_a == test_label).sum().item()
            total = test_label.size(0)
            
            test_loss.append(loss.item())
            test_acc.append(100 * correct / total)
        avg_test_loss = sum(test_loss) / len(test_loss)
        avg_test_acc = sum(test_acc) / len(test_acc)
    return avg_test_acc, avg_test_loss
        
    
     

if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    
    # args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = args_parser()
    logger = setup_logger(args)
    logger.info(device)
    logger.info("命令行参数:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    train(args,logger)
    
