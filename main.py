import pandas as pd
import torch
from lib.data_example import *
from lib.options import args_parser
from random import choice
from lib.models.models import *
from lib.models.models_cifar import *
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
from lib.Class_incremental import *
from lib.Feature_incremental import *



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
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    if args.class_type == "class_task":
        Class_train(args, logger)
    elif args.class_type == "feature_task":
        Feature_train(args, logger)
