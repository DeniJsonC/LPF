import os
import random
import time

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.losses
from dataset import DataLoaderTrain, DataLoaderVal
from model.Mix_Aug import Mixing_Augment
from model.LPF import LPF
from utils import losses, lr_scheduler, network_parameters

## Set Seeds
GLOBALSEED=5
torch.backends.cudnn.benchmark = True
random.seed(GLOBALSEED)
np.random.seed(GLOBALSEED)
torch.manual_seed(GLOBALSEED)
torch.cuda.manual_seed_all(GLOBALSEED)

## Load yaml configuration file
with open('config/RRNet-lol.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']
Dataset=opt['DATASET']
Model=opt['MODEL']

if Train['SCALE_TRAINING']:
    scale_t=opt['SCALE_TRAINING']
    train_scale_list=[]
    val_scale_list=[]
    sort=1
    if scale_t['SORT']==0:
        for i in range(scale_t['BASE'],scale_t['BASE']+scale_t['SCALE']*scale_t['STEP']+1,scale_t['SCALE']):
            train_scale_list.append(i)
            #val_scale_list.append(Dataset['VAL_PS'][0])
        scale_t['BATCH'].sort(reverse=True)
        OPT['BATCH']=scale_t['BATCH']
    else:
        for i in range(scale_t['BASE']+scale_t['SCALE']*scale_t['STEP'],scale_t['BASE']-1,-scale_t['SCALE']):
            train_scale_list.append(i)
        OPT['BATCH']=scale_t['BATCH']
    Dataset['TRAIN_PS']=train_scale_list
    OPT['TOTAL_EPOCHS']=scale_t['TOTAL_EPOCHS']
## Build Model
print('==> Build the model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_restored=LPF()
p_number = network_parameters(model_restored)
model_restored.to(device)

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Dataset['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Dataset['TRAIN_DIR']
val_dir = Dataset['VAL_DIR']

## GPU
if device == 'cuda':
    gpus = ','.join([str(i) for i in opt['GPU']])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    if len(device_ids) > 1:
        model_restored = nn.DataParallel(model_restored, device_ids=device_ids)
else:
    device_ids=None

## Optimizer
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.AdamW(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999),weight_decay=1e-4)
## Scheduler (Strategy)
scheduler=lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer,periods=[int(OPT['TOTAL_EPOCHS']*0.45),int(OPT['TOTAL_EPOCHS']*0.77)],
                                                      restart_weights=[1,1],eta_mins=[OPT['LR_INITIAL'],OPT['LR_MIN']])
# scheduler=lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer,periods=[int(OPT['TOTAL_EPOCHS']*0.45),int(OPT['TOTAL_EPOCHS']*0.55)],
#                                                         restart_weights=[1,1],eta_mins=[OPT['LR_INITIAL'],OPT['LR_MIN']])
total_epochs=1

# Start training!

best_psnr = 0
p_ssim=0
best_ssim = 0
s_psnr=0
watch_epoch=0
best_score=0
best_epoch_psnr = 0
best_epoch_ssim = 0

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    total_epochs = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, total_epochs):
        scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
uloss=losses.ULoss()

total_start_time = time.time()

## Log
log_dir = os.path.join(Dataset['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')
print('==>Scale Training start: ')
while total_epochs<=OPT['TOTAL_EPOCHS']:
    for j in range(len(OPT['BATCH'])):
        if total_epochs>OPT['TOTAL_EPOCHS']:
            break
        if Train['SCALE_TRAINING']:
            if scale_t['CROP']:               
                for m in range(2):
                    if total_epochs>OPT['TOTAL_EPOCHS']:
                        break
                    scale_t['WATCH_EPOCHS']+=1
                    if m%2==0:
                        Dataset['CROP']=True
                    else:
                        Dataset['CROP']=False
                    print('==> Loading datasets')
                    val_loader = DataLoader(DataLoaderVal(Dataset['VAL_DIR'],img_options=Dataset,), batch_size=1 ,shuffle=False, num_workers=0)
                    train_loader = DataLoader(DataLoaderTrain(Dataset['TRAIN_DIR'],bs_j=j,img_options=Dataset), 
                                                batch_size=OPT['BATCH'][j], shuffle=False, num_workers=0)
                                                # Show the training configuration
                    print(f'''==> Training details:
                    ------------------------------------------------------------------
                        Restoration mode:   {mode}
                        Train patches size: {str(Dataset['TRAIN_PS'][j]) + 'x' + str(Dataset['TRAIN_PS'][j])}
                        Val patches size:   {str(Dataset['VAL_PS'][0]) + 'x' + str(Dataset['VAL_PS'][0])}
                        Model parameters:   {str(p_number/1e6)+'M'}
                        Start/End epochs:   {str(total_epochs) + '~' + str(OPT['TOTAL_EPOCHS'])}
                        Scale training:     {Train['SCALE_TRAINING']}
                        Batch sizes:        {OPT['BATCH'][j]}
                        Crop states:        {Dataset['CROP']}
                        Learning rate:      {scheduler.get_last_lr()[0]}
                        device:             {device}
                        watch dog:          {scale_t['WATCH_EPOCHS']}
                        GPU:                {'GPU' + str('NULL' if device_ids is None else device_ids)}''')
                    print('------------------------------------------------------------------')
                    for epoch in range(total_epochs, OPT['TOTAL_EPOCHS'] + 1):
                        epoch_start_time = time.time()
                        epoch_loss = 0
                        train_id = 1
                        for i, data in enumerate(tqdm(train_loader), 0):
                            # Forward propagation
                            input_ = data[0].to(device)
                            target = data[1].to(device)
                            if Dataset['MIX_AUG']:
                                target,input_=Mixing_Augment(device)(target,input_)
                            restored=model_restored(input_)

                            # Back propagation

                            loss=uloss(restored,target)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model_restored.parameters(), 0.01)
                            optimizer.step()
                            optimizer.zero_grad()
                            epoch_loss += loss.item()
                        ## Evaluation (Validation)
                        if total_epochs % Train['VAL_AFTER_EVERY'] == 0:
                            psnr_val_rgb = []
                            ssim_val_rgb = []
                            images_list=[]
                            fn_list=[]
                            for ii, data_val in enumerate(val_loader, 0):
                                input_ = data_val[0].to(device)
                                target = data_val[1].to(device)
                                file_name=data_val[2][0]
                                with torch.no_grad():
                                    restored= model_restored(input_)
                                    psnr_val_rgb.append(utils.torchPSNR(restored, target))
                                    ssim_val_rgb.append(utils.torchSSIM(restored, target))
                                    images_list.append(restored)
                                    fn_list.append(file_name)
                            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
                            #svae best score restored images
                            if ssim_val_rgb*10+psnr_val_rgb>best_score:
                                best_score=ssim_val_rgb*10+psnr_val_rgb
                                watch_epoch=0
                                if Train['SAVE_VAL_RESULTS']:
                                    for img,fn in zip(images_list,fn_list):
                                        pred = (np.clip(img[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
                                        imageio.imwrite(os.path.join(Train['SAVE_DIR'],'{}.png'.format(fn)), pred)
                                    del img,pred
                                del images_list,fn_list
                            else:
                                watch_epoch+=1
                            # Save the best PSNR model of validation
                            if psnr_val_rgb > best_psnr:
                                best_psnr = psnr_val_rgb
                                p_ssim=ssim_val_rgb
                                best_epoch_psnr = total_epochs
                                torch.save({'epoch': total_epochs,
                                            'state_dict': model_restored.state_dict(),
                                            'optimizer': optimizer.state_dict()
                                            }, os.path.join(model_dir, "model_bestPSNR_{}_{}_{}.pth".format(best_epoch_psnr,best_psnr,ssim_val_rgb,2)))
                            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f ||SSIM %.4f ||WatchDog %d]" % (
                                total_epochs, psnr_val_rgb, best_epoch_psnr, best_psnr,p_ssim,watch_epoch))
                            # Save the best SSIM model of validation
                            if ssim_val_rgb > best_ssim:
                                best_ssim = ssim_val_rgb
                                s_psnr=psnr_val_rgb
                                best_epoch_ssim = total_epochs
                                torch.save({'epoch': total_epochs,
                                            'state_dict': model_restored.state_dict(),
                                            'optimizer': optimizer.state_dict()
                                            }, os.path.join(model_dir, "model_bestSSIM_{}_{}_{}.pth".format(best_epoch_ssim,best_ssim,psnr_val_rgb)))
                            print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f ||PSNR %.4f ||WatchDog %d]" % (
                                total_epochs, ssim_val_rgb, best_epoch_ssim, best_ssim,s_psnr,watch_epoch))
                            """
                            # Save evey epochs of model
                            torch.save({'epoch': epoch,
                                        'state_dict': model_restored.state_dict(),
                                        'optimizer': optimizer.state_dict()
                                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
                            """
                            writer.add_scalar('val/PSNR', psnr_val_rgb, total_epochs)
                            writer.add_scalar('val/SSIM', ssim_val_rgb, total_epochs)
                        scheduler.step()
                        print("------------------------------------------------------------------")
                        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(total_epochs, time.time() - epoch_start_time,
                                                                                                epoch_loss, scheduler.get_last_lr()[0]))
                        print("------------------------------------------------------------------")
                        # Save the last model
                        torch.save({'epoch': total_epochs,
                                    'state_dict': model_restored.state_dict(),
                                    'optimizer': optimizer.state_dict()
                                    }, os.path.join(model_dir, "model_latest.pth"))
                        writer.add_scalar('train/loss', epoch_loss, total_epochs)
                        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], total_epochs)
                        total_epochs+=1

                        if total_epochs>OPT['TOTAL_EPOCHS']:
                            break
                        if watch_epoch>0 and watch_epoch%scale_t['WATCH_EPOCHS']==0:
                            watch_epoch=0
                            break
writer.close()


total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
