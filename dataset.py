import os
from torch.utils.data import Dataset
import os.path as osp
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
import yaml
import imageio
from torchvision import transforms
import torch.nn.functional as F
from model.Mix_Aug import Mixing_Augment

SEED=5
torch.backends.cudnn.benchmark = True
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, bs_j,img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target
        self.data_aug=self.img_options['DATA_AUG']
        self.crop=self.img_options['CROP']
        #self.transfom=self.img_options['TRANSFORM']
        self.ps = self.img_options['TRAIN_PS'][bs_j]

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, ( 0,padw, 0, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, ( 0,padw, 0, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        if self.crop:
            hh, ww = tar_img.shape[1], tar_img.shape[2]

            rr = random.randint(0, hh - ps)
            cc = random.randint(0, ww - ps)


            # Crop patch
            inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
            tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if self.data_aug:
            aug = random.randint(0, 8)
            if aug == 1:
                inp_img = inp_img.flip(1)
                tar_img = tar_img.flip(1)
            elif aug == 2:
                inp_img = inp_img.flip(2)
                tar_img = tar_img.flip(2)
            elif aug == 3:
                inp_img = torch.rot90(inp_img, dims=(1, 2))
                tar_img = torch.rot90(tar_img, dims=(1, 2))
            elif aug == 4:
                inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
                tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            elif aug == 5:
                inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
                tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            elif aug == 6:
                inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
                tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            elif aug == 7:
                inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
                tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
        # if self.transfom:
        #     t=transforms.ColorJitter(brightness=self.img_options['BRIGHTNESS'],
        #                                 contrast=self.img_options['CONTRAST'],
        #                                 saturation=self.img_options['SATURATION'])
        #     inp_img=t(inp_img)
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return transforms.Resize((self.ps,self.ps))(inp_img), transforms.Resize((self.ps,self.ps))(tar_img),filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir,img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.mc_size=img_options['MATCH_SIZE']
        self.sizex = len(self.tar_filenames)  # get the size of target
        self.resize=self.img_options['VAL_RESIZE']
        self.ps = self.img_options['VAL_PS'][0]
        self.crop=self.img_options['VAL_CROP']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]        
        # Validate on center crop or rescale
        if self.crop:
            #center-crop
            inp_img =transforms.CenterCrop(ps)(inp_img)
            tar_img =transforms.CenterCrop(ps)(tar_img)

            #rescale
        if self.resize:
            #center-crop
            # inp_img =transforms.CenterCrop(ps)(inp_img)
            # tar_img =transforms.CenterCrop(ps)(tar_img)

            #rescale
            
            inp_img = TF.resize(inp_img, ps)
            tar_img = TF.resize(tar_img, ps)
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        if self.mc_size:
            inp_img=self.match_size(inp_img)
            tar_img=self.match_size(tar_img)
        return inp_img, tar_img,filename

            # Pad the input if not_multiple_of 8




    def match_size(self,input_,mul=32):
        h, w = input_.shape[1], input_.shape[2]
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        input_ = F.pad(input_, ( 0,padw, 0, padh), 'reflect')
        return input_


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')

        inp = TF.to_tensor(inp)
        return inp, filename


    
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     with open('training.yaml', 'r') as config:
#         opt = yaml.safe_load(config)
#     dataset = DataLoaderTrain("../dataset/LOLdataset/eval15/",opt['DATASET'])
#     print("数据个数：", len(dataset))
#     train_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                 batch_size=1, 
#                                                shuffle=False)
#     save_images='test_images'
#     for epoch in range(3):
#         print('epoch:',epoch)
#         for image_num,img in enumerate(train_loader):
#             img[1],img[0]=Mixing_Augment(device,use_identity=True)(img[1],img[0])
#             file_name=img[2][0]
#             pred = (np.clip(img[0].detach().cpu().numpy().transpose(0,2,3,1),0,1)*255).astype(np.uint8)
#             gt = (np.clip(img[1].detach().cpu().numpy().transpose(0,2,3,1),0,1)*255).astype(np.uint8) 
#             imageio.imwrite(os.path.join(save_images,'pred_img_num_{}.png'.format(file_name)), pred[0,:,:,:])
#             imageio.imwrite(os.path.join(save_images,'{}.png'.format(file_name)), gt[0,:,:,:])
