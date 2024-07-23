import os
import random
from glob import glob

import numpy as np
from PIL import Image
from numpy import asarray
import pandas as pd
import torch
from skimage import transform
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import cv2
import math
class Resize:
    """Resize the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # print(sample)
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        sample['image'] = transform.resize(sample['image'], (new_h, new_w), preserve_range=True).astype('uint8')
        #ntaurus
        # sample['depth_image'] = transform.resize(sample['depth_image'], (new_h, new_w), preserve_range=True).astype('uint8')
        return sample


class CenterCrop2:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        sample['image'] = functional.center_crop(sample['image'], self.output_size)
        #ntaurus
        # sample['depth_image'] = functional.center_crop(sample['depth_image'], self.output_size)

        return sample


class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability
        self.flip = transforms.RandomHorizontalFlip(p=probability)

    def __call__(self, sample):
        sample['image'] = self.flip(sample['image'])
        return sample


class RandomVerticalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability
        self.flip = transforms.RandomVerticalFlip(p=probability)

    def __call__(self, sample):
        sample['image'] = self.flip(sample['image'])
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #ntaurus depth_image
        # to_tensor = transforms.ToTensor()
        # depth = sample['depth_image']
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        # depth = depth / 65535
        # depth = to_tensor(depth).float()

        sample['image'] = torch.from_numpy(sample['image'].transpose((2, 0, 1)).astype(float))
        # sample['image'] = torch.cat((sample['image'], depth), 0)
        
        keys = set(sample.keys()).difference(['image'])
        for key in keys:
            sample[key] = torch.from_numpy(sample[key])
        return sample


val_MAX = [3943.325195, 7974, 106.343002, 844.568604, 147.491821]
K = 1
class Normalize:
    """Normalize values."""

    def __init__(self, image_means, image_stds):
        self.means = image_means
        self.stds = image_stds
        self.calories_max = val_MAX[0]
        self.mass_max = val_MAX[1]
        self.fat_max = val_MAX[2]
        self.carb_max = val_MAX[3]
        self.protein_max = val_MAX[4]

    def __call__(self, sample):
        # sample['mass'] = sample['mass']         / self.mass_max * K
        # sample['calorie'] = sample['calorie'] / self.calories_max * K
        # sample['fat'] = sample['fat'] / self.fat_max * K
        # sample['carb'] = sample['carb'] / self.carb_max * K
        # sample['protein'] = sample['protein'] / self.protein_max * K

        # sample['calorie'] = math.log(sample['calorie'], math.e)
        # sample['mass'] = math.log(sample['mass'], math.e)
        # sample['fat'] = math.log(sample['fat'], math.e)
        # sample['carb'] = math.log(sample['carb'], math.e)
        # sample['protein'] = math.log(sample['protein'], math.e)


        sample['image'] = functional.normalize(sample['image'], self.means, self.stds)
        return sample



def create_nutrition_df(root_dir, sampling_rate=5):
    csv_file = os.path.join(root_dir, 'metadata', 'dish_metadata_cafe1_after_clear.csv')
    print(csv_file)
    dish_metadata = {'dish_id': [], 'mass': [], 'calorie': [], 'fat': [], 'carb': [], 'protein': [], 'frame': []}
    # dish_metadata = {'dish_id': [], 'mass': [], 'calorie': [], 'fat': [], 'carb': [], 'protein': [], 'frame': [], 'depth_image':[]}
    with open(csv_file, "r") as f:
        for line in f.readlines():
            parts = line.split(',')

            dish_id = parts[0]

            #ntaurus
#            frames_path = os.path.join(root_dir, 'side_angles',
#                                       dish_id, 'camera_A')

#            if not os.path.isdir(dish_path):
#                continue
#            image_path = dish_path + '/rgb.png'
#            if not os.path.isfile(image_path):
#                continue
##            for i in range(50):
#            dish_metadata['dish_id'].append(parts[0])
#            dish_metadata['calorie'].append(int(float(parts[1])))
#            dish_metadata['mass'].append(parts[2])
#            dish_metadata['fat'].append(parts[3])
#            dish_metadata['carb'].append(parts[4])
#            dish_metadata['protein'].append(parts[5])
#            dish_metadata['frame'].append(image_path)
#
            # frames_path = os.path.join(root_dir, 'images',dish_id)
            # depth_image_path = frames_path + '/depth_color.png'
            # rgb_image_path = frames_path + '/rgb.png'
            # rgb_frame_path = os.path.join(root_dir, 'side_angles', dish_id, 'camera_A', '5.jpg')
            # if not os.path.isfile(rgb_image_path):
                # continue
                
            # if not os.path.isfile(depth_image_path):
                # continue
     
            # if not os.path.isfile(rgb_frame_path):
                # continue
            # print(rgb_frame_path)
            # dish_metadata['dish_id'].append(parts[0])
            # dish_metadata['calorie'].append(int(float(parts[1])))
            # dish_metadata['mass'].append(parts[2])
            # dish_metadata['fat'].append(parts[3])
            # dish_metadata['carb'].append(parts[4])
            # dish_metadata['protein'].append(parts[5])
            # dish_metadata['frame'].append(rgb_image_path)
            # dish_metadata['depth_image'].append(depth_image_path)
            
            # dish_metadata['dish_id'].append(parts[0])
            # dish_metadata['calorie'].append(int(float(parts[1])))
            # dish_metadata['mass'].append(parts[2])
            # dish_metadata['fat'].append(parts[3])
            # dish_metadata['carb'].append(parts[4])
            # dish_metadata['protein'].append(parts[5])
            # dish_metadata['frame'].append(rgb_frame_path)
            # dish_metadata['depth_image'].append(depth_image_path)

#            frames_path = os.path.join(root_dir, 'imagery', 'side_angles',
#                                       dish_id,
#                                       'frames')

            camera_files = ['camera_A', 'camera_B', 'camera_C', 'camera_D']
            # camera_files = ['camera_A', '', '', '']
            for i in range(4):
                # print(camera)
            # if not os.path.isdir(frames_path):
                # continue
                frames_path = os.path.join(root_dir, 'side_angles',
                                      dish_id, camera_files[i])
                # print(frames_path)
                if not os.path.isdir(frames_path):
                    continue
                frames = sorted(glob(frames_path + os.path.sep + '*.jpg'))
                # print(frames)
                # print("1111")
                for i, frame in enumerate(frames):
                    if i % sampling_rate == 0:
                        dish_metadata['dish_id'].append(parts[0])
                        dish_metadata['calorie'].append((float(parts[1])))
                        dish_metadata['mass'].append(float(parts[2]))
                        dish_metadata['fat'].append(float(parts[3]))
                        dish_metadata['carb'].append(float(parts[4]))
                        dish_metadata['protein'].append(float(parts[5]))
                        dish_metadata['frame'].append(frame)
                        # dish_metadata['depth_image'].append(depth_image_path)
#             img_d = glob(root_dir + os.path.sep + images + os.path.sep + dish_id + os.path.sep + 'depth_color.png')
#            img_d = os.path.join(root_dir, 'images', dish_id, 'depth_color.png')
#            if not os.path.isfile(img_d):
#                continue
#            dish_metadata['dish_id'].append(parts[0])
#            dish_metadata['calorie'].append(int(float(parts[1])))
#            dish_metadata['mass'].append(parts[2])
#            dish_metadata['fat'].append(parts[3])
#            dish_metadata['carb'].append(parts[4])
#            dish_metadata['protein'].append(parts[5])
#            dish_metadata['frame'].append(img_d)
#
#            
#            dish_metadata['dish_id'].append(parts[0])
#            dish_metadata['calorie'].append(int(float(parts[1])))
#            dish_metadata['mass'].append(parts[2])
#            dish_metadata['fat'].append(parts[3])
#            dish_metadata['carb'].append(parts[4])
#            dish_metadata['protein'].append(parts[5])
#            dish_metadata['frame'].append(img_d)

             
    return pd.DataFrame.from_dict(dish_metadata)


def split_dataframe(dataframe: pd.DataFrame, split):
    dish_ids = dataframe.dish_id.unique()
    random.shuffle(dish_ids)
    train_end = int(len(dish_ids) * split['train'])
    train_ids = dish_ids[:train_end]
    train_df = dataframe[dataframe['dish_id'].isin(train_ids)]
    train_index = list(train_df.index.copy())
    random.shuffle(train_index)
    train_df = train_df.loc[train_index]

    val_end = train_end + int(len(dish_ids) * split['validation'])
    val_ids = dish_ids[train_end:val_end]
    val_df = dataframe[dataframe['dish_id'].isin(val_ids)]
    val_index = list(val_df.index.copy())
    random.shuffle(val_index)
    val_df = val_df.loc[val_index]

    test_ids = dish_ids[val_end:]
    test_df = dataframe[dataframe['dish_id'].isin(test_ids)]
    test_index = list(test_df.index.copy())
    random.shuffle(test_index)
    test_df = test_df.loc[test_index]

    print("trainset len is {}".format(len(train_df)))
    print("valset len is {}".format(len(val_df)))
    print("testset len is {}".format(len(test_df)))

    return train_df, val_df, test_df


def to_ndarray(value):
    value = np.array([value])
    return value.astype('float').reshape(1, 1)


class Nutrition5kDataset(Dataset):
    def __init__(self, dish_metadata, root_dir, transform=None):
        self.dish_metadata = dish_metadata
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dish_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.dish_metadata.iloc[idx]['frame']
	
        image = Image.open(frame).convert('RGB')
        image = asarray(image)
        # image = image[ 0: -300, 230 : -500]
		# ntaurus depth_model
        # depth_image = Image.open(self.dish_metadata.iloc[idx]['depth_image'])
        # depth_image = asarray(depth_image)
#        print(image)
#        print("------------------------")
#        print(depth_image)
		
        calories = to_ndarray(self.dish_metadata.iloc[idx]['calorie'])
        mass = to_ndarray(self.dish_metadata.iloc[idx]['mass'])
        fat = to_ndarray(self.dish_metadata.iloc[idx]['fat'])
        carb = to_ndarray(self.dish_metadata.iloc[idx]['carb'])
        protein = to_ndarray(self.dish_metadata.iloc[idx]['protein'])

        sample = {'image': image, 'mass': mass, 'fat': fat, 'carb': carb, 'protein': protein, 'calorie': calories}
        # sample = {'image': image, 'mass': mass, 'fat': fat, 'carb': carb, 'protein': protein, 'calorie': calories, 'depth_image': depth_image}
        if self.transform:
            sample = self.transform(sample)

        return sample
