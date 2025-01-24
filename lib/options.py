'''
Author: wangshuo105 3220215214@bit.edu.cn
Date: 2023-10-13 20:05:55
LastEditors: wangshuo105 3220215214@bit.edu.cn
LastEditTime: 2024-03-02 14:22:53
FilePath: /Heterogeneous_vertical_federated_learning/lib/options.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # VFL
    parser.add_argument('--global_epoch', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_user', type=int, default=4, help="number of party(both active and passive): K")
    parser.add_argument('--frac', type=float, default=0.04, help='the fraction of clients: C')
    parser.add_argument('--local_bs', type=int, default=4, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--baseline_type',type=str,default="Avg",help="Avg,Con,Max")
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--train_type", type=str, default="homo",help="train type:heter,homo")
    parser.add_argument("--task_type", type=str, default="class_enhance",help="train type:feature_enhance,class_enhance")
    parser.add_argument("--task", type=int, default=4,help="the number of task")
    parser.add_argument("--lambda_proto_aug", type=float, default=0.5,help="class_enhance loss hyper_paramerse")
    parser.add_argument("--r", type=float, default=0, help="class_enhance noise hyper_paramerse")
    parser.add_argument("--K", type=float, default=0.5,help="loss function")
    parser.add_argument("--radius", type=float, default=0.01,help="create noise")
    parser.add_argument("--Abaltion_LCE", type=int, default=1,help="ablation need LCE")
    parser.add_argument("--Abaltion_LA", type=int, default=1,help="ablation need LCE")
    parser.add_argument("--Abaltion_LMO", type=int, default=1,help="ablation need LCE")
    parser.add_argument("--class_type",type=str,default="class_task",help="whether or class_task")
    
    parser.add_argument("--k_0", type=float, default=15,help="create noise")
    parser.add_argument("--alpha", type=float, default=3,help="create noise")
    # parser.add_argument("--folder_name", type=str, default="",help="create noise")
    
    # parser.add_argument("--exper_number",type=str,default="E1", help="experiment number")
    # parser.add_argument("--each_batch_shapley", type=int, default=0, help="whether compute the contribution by each batch size")
    parser.add_argument("--weight_type", type=str, default="global_weight",help="wheter using normalization")
    # 第三个实验，采用贡献值聚合的方法(global_weight,current_weight,no_normalize)
    parser.add_argument("--history_rate",type=float,default=0.3,help="history contribution rate")
    parser.add_argument('--model', type=str, default='lenet', help="model name")
    parser.add_argument('--mode', type=str, default='task_heter', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--out_channels', type=int, default=16, help="out of channels of imgs")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--ep', type=str, default=1,help="ep number")
    # data
    parser.add_argument('--datasets',type=str, default='Fmnist', help="dataset type")
    #parser.add_argument('--train', type=str, default='True', help="the dataset is or not training")
    # parser.add_argument('--split', type=int, default=392, help="split imgs feature")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch_size")
    parser.add_argument('--shuffle', type=str, default='True',help="whether shuffle the dataset")
    parser.add_argument('--nThreads',type=int, default=10, help="number of threads when read data")
    # other
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="name of classes")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--seed', type=int, default=1234, help="random seed")
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")
    parser.add_argument('--agg', type=str, default='False', help="whether use the agg out")
    parser.add_argument('--optim', type=str, default='False', help="whether use the agg out")

    args = parser.parse_args()
    return args

