import time
import logging
import os
from glob import glob
from shutil import copyfile
import datetime as dt
from nutrition5k.nutritionModel import create_model
import torch.nn as nn

'''
DEBUG_MODE == True is intended for testing code, while 
DEBUG_MODE == False is intended to enable faster training and inference
'''
DEBUG_MODE = True
import numpy as np
import torch
CUDA_VERSION = torch.version.cuda
logging.warning('cuda version: {}'.format(CUDA_VERSION))
'''
if CUDA_VERSION:
    logging.warning('CUDA_PATH: {}'.format(os.environ['CUDA_PATH']))
    logging.warning('CUDA_HOME: {}'.format(os.environ['CUDA_HOME']))
'''

if DEBUG_MODE:
    np.random.seed(0)
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    torch.autograd.profiler.profile(True)
    torch.backends.cudnn.benchmark = False
else:
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
torch.set_printoptions(linewidth=120)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from nutrition5k import n5kloss
from nutrition5k import swinNLoss
from nutrition5k.dataset import Resize, ToTensor, CenterCrop2, RandomHorizontalFlip, RandomVerticalFlip, Normalize, \
    Nutrition5kDataset, create_nutrition_df, split_dataframe
from nutrition5k.model import Nutrition5kModel
from nutrition5k.train_utils import run_epoch_ensemble
from nutrition5k.utils import parse_args, Metrics


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
SECONDS_TO_HOURS = 3600
IMAGE_RESOUTION = (250, 250)
CROP_SIZE = (224, 224)

def create_dataloaders():
    nutrition_df = create_nutrition_df(config['dataset_dir'])
    train_df, val_df, test_df = split_dataframe(nutrition_df, config['split'])

    train_set = Nutrition5kDataset(train_df, config['dataset_dir'], transform=transforms.Compose(
        [Resize(IMAGE_RESOUTION),
        ToTensor(),
        CenterCrop2(CROP_SIZE),

        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    val_set = Nutrition5kDataset(val_df, config['dataset_dir'], transform=transforms.Compose(
        [Resize(IMAGE_RESOUTION),
        ToTensor(),
        CenterCrop2(CROP_SIZE),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    test_set = Nutrition5kDataset(test_df, config['dataset_dir'], transform=transforms.Compose(
        [Resize(IMAGE_RESOUTION),
        ToTensor(),
        CenterCrop2(CROP_SIZE),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    #ntaurus depth_image
    # train_set = Nutrition5kDataset(train_df, config['dataset_dir'], transform=transforms.Compose(
        # [Resize(IMAGE_RESOUTION),
         # ToTensor(),
         # Normalize([0.485, 0.456, 0.406, 0.458], [0.229, 0.224, 0.225, 0.228])]))

    # val_set = Nutrition5kDataset(val_df, config['dataset_dir'], transform=transforms.Compose(
        # [Resize(IMAGE_RESOUTION),
         # ToTensor(),
         # Normalize([0.485, 0.456, 0.406, 0.458], [0.229, 0.224, 0.225, 0.228])]))

    # test_set = Nutrition5kDataset(test_df, config['dataset_dir'], transform=transforms.Compose(
        # [Resize(IMAGE_RESOUTION),
         # ToTensor(),
         # Normalize([0.485, 0.456, 0.406, 0.458], [0.229, 0.224, 0.225, 0.228])]))
    return {
        'train': DataLoader(train_set, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['dataset_workers'], pin_memory=True),
        'val': DataLoader(val_set, batch_size=config['batch_size'], shuffle=False,
                          num_workers=config['dataset_workers'], pin_memory=True),
        'test': DataLoader(test_set, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['dataset_workers'], pin_memory=True)
    }

def load_pretrained(model = None, path = ""):
    # model_dict = model.state_dict()  # 取出自己网络的参数字典
    # # print(model_dict)
    # print(path)
    # pretrained_dict = torch.load(path)  # 加载预训练网络的参数字典
    # print(pretrained_dict)

    net_dict = model.state_dict()
    pretrained_model = torch.load(path, map_location='cuda:0')
    state_dict = {}

    i = 0
    for K, dict_ in pretrained_model.items():
        # print(dict_)
        for k, v in dict_.items():
            k = k.replace('position', 'positive')
            k = "base_model." + k
            # print(k)
            # print(v)
            if k in net_dict.keys() and "head" not in k:
                # print(v.requires_grad)
                # 4.17 冻结
                # if i < 130:
                #     v.requires_grad = False
                # else:
                v.requires_grad = True
                state_dict[k] = v
                # print(k)
                print(v.requires_grad)
                # print("--------------------------------")
            i += 1
        break
    print(i)
    net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
    model.load_state_dict(net_dict)
    # print("+++++++++++++++++++++++++++++++++++++++")
    # print(model.state_dict())
    return model

# nohup python train.py > /home/NLM/myProgram/Log/myout.log 2>&1  &
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    with open(args.config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)
    comment = f'batch_size = {config["batch_size"]} lr = {config["learning_rate"]}'
    log_dir = os.path.join(config['log_dir'], config['experiment_name'], str(dt.datetime.now()))
    tensorboard = SummaryWriter(comment=comment, log_dir=log_dir)

    copyfile(args.config_path, os.path.join(tensorboard.log_dir, os.path.basename(args.config_path)))

    if config['start_checkpoint']:
        dataloaders = torch.load(os.path.join(config['start_checkpoint'], 'dataloaders.pt'))
    else:
        dataloaders = create_dataloaders()
        torch.save(dataloaders, os.path.join(tensorboard.log_dir, 'dataloaders.pt'))

    epoch_phases = ['train']
    if len(dataloaders['val']) > 0:
        epoch_phases.append('val')
    # print(epoch_phases)
    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    devices = [device, device2]
    #model = Nutrition5kModel(config['task_list']).float().to(device)
    # model = Swin_Nutrition(config['task_list']).float().to(device)
    model = create_model(backone="inception_v4", tasks=config["task_list"], device=device, fc_d=2048)
    model2 = create_model(backone="mvitv2_tiny", tasks=config["task_list"], device=device2, fc_d=2048)
    # botnet26t_256
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M):%.2f' % (n_parameters / 1.e6))

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=config['learning_rate'])

    # criterion = n5kloss
    criterion = swinNLoss
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=config['lr_scheduler']['patience'])
    best_model_path = None

    since = time.time()
    best_val_loss = np.inf
    best_training_loss = np.inf
    issave = 0
    for epoch in tqdm(range(config['epochs'])):
        training_loss = None
        val_loss = None
        optimizer.zero_grad(set_to_none=True)
        for phase in epoch_phases:
            metrics = Metrics(config['task_list'], device, config['prediction_threshold'])

            if phase == 'train':
                model.train()
                model2.train()
            else:
                model.eval()
                model2.eval()
            results = run_epoch_ensemble(model, model2, criterion, dataloaders[phase], devices, phase,
                                config['mixed_precision_enabled'], optimizer=optimizer, optimizer2=optimizer2,
                                lr_scheduler=lr_scheduler,
                                gradient_acc_steps=config['gradient_acc_steps'],
                                lr_scheduler_metric=config['lr_scheduler']['metric'],
                                task_list=config['task_list'],
                                 metrics=metrics)
            if phase == 'train':
                training_loss = results['average loss']
            else:
                val_loss = results['average loss']

            metrics_results = metrics.compute()
            # metrics.reset()

            tensorboard.add_scalar('{} loss'.format(phase), results['average loss'], epoch)
            print('Epoch {} {} loss: {:.4f}'.format(epoch, phase, results['average loss']))

            for task, metric_value in metrics_results.items():
                metric_name = '{} {}'.format(phase, task)
                print('Epoch {} {}: {:.4f}'.format(epoch, metric_name, metric_value))
                tensorboard.add_scalar(metric_name, metric_value, epoch)

        if val_loss and (val_loss < best_val_loss) or (not config['save_best_model_only']):
            epoch_dir = os.path.join(tensorboard.log_dir, 'epoch_{}'.format(epoch))
            os.makedirs(epoch_dir, exist_ok=True)
            state_dict = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            torch.save(state_dict, epoch_dir + "/state_dict_model1_" + str(epoch) + ".pth")

            state_dict2 = {"net": model2.state_dict(), "optimizer": optimizer2.state_dict(), "epoch": epoch}
            torch.save(state_dict2, epoch_dir + "/state_dict_model2_" + str(epoch) + ".pth")

            if not issave:
                issave = 1
                torch.save(model, epoch_dir + "/model.pth")
                torch.save(model2, epoch_dir + "/model2.pth")

            best_val_loss = val_loss
        if training_loss < best_training_loss:
            best_training_loss = training_loss

        time_elapsed = time.time() - since
        if config['max_training_time'] and (time_elapsed // SECONDS_TO_HOURS) > config['max_training_time']:
            print('Time limit exceeded. Stopping training. ')
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best training loss is {:.4f}'.format(best_training_loss))
    tensorboard.add_hparams(
        {'learning rate': config['learning_rate'], 'batch size': config['batch_size']},
        {
            'best training loss': best_training_loss,
            'best validation loss': best_val_loss
        },
    )
    tensorboard.close()
