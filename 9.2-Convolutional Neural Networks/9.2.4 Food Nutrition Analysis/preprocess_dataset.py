import argparse
import os

from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image


def parse_args():
    """ Parse the arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path', default="G:\LoadDown datasets\\nutrition5k_dataset\\nutrition5k_dataset-2\\nutrition5k_dataset",
                        help='Path to Nutrition5k dataset.')
    return parser.parse_args()

from torchvision import transforms
import torch
import numpy as np

if __name__ == '__main__':


    scale = (480, 640)

    resize_img = transforms.Resize(scale, Image.BILINEAR)
    resize_depth = transforms.Resize(scale, Image.NEAREST)
    to_tensor = transforms.ToTensor()

    img_id = 0

    # load image and resize
    img = Image.open('/home/NLM/myProgram/nutrition5k-main/dataset/images/dish_1556572657/' + 'rgb.png')
    img = resize_img(img)
    img = np.array(img)

    # load depth and resize
    # depth = Image.open('/home/NLM/myProgram/nutrition5k-main/dataset/images/dish_1556572657/' + 'depth_color.png')
    # depth = resize_depth(depth)
    # depth = np.array(depth)

    depth = cv2.imread(r'/home/NLM/myProgram/nutrition5k-main/dataset/images/dish_1556572657/depth_color.png') # 填要转换的图片存储地址
    # print(depth[:][:][1])
    
    # depth = depth[:][:][0]
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite(r'C:\Users\room8.png',depth) # 填转换后的图片存储地址，若在同一目录，则注意不要重名
    # print(depth.shape)
    # depth = np.array(depth)
    # depth = torch.from_numpy(depth.transpose((2, 0, 1)).astype(float))
    # print(depth.shape)
    # print(depth[0][:][:])
    
    # for x in range(480):
        # for y in range(640):
            # if depth[1][x][y] != 0:
            # print(depth[1][x][y], end="")
        # print()
    
    
    # print("```````````````````````")
    # print(depth[1][:][:])
    # print("```````````````````````")    
    # print(depth[2][:][:])

    # depth = depth[:, :, np.newaxis]
    
    # ori shape and value
    # print(img.shape)
    # print(depth.shape)
    # print(img[2][2][0])  # 215
    # print(img[2][2][1])  # 230
    # print(img[2][2][2])  # 227
    # print(depth[2][2][0])  # 16664

    # tensor shape and value, normalization
    img = Image.fromarray(img).convert('RGB')
    img = to_tensor(img).float()
    print(img.shape)
    # print(img[0][2][2])  # tensor(0.8431)
    # print(img[1][2][2])  # tensor(0.9020)
    # print(img[2][2][2])  # tensor(0.8902)

    depth = depth / 65535
    depth = to_tensor(depth).float()
    # print(depth.shape)
    # print(depth[0][2][2])  # tensor(0.2543)

    rgbd = torch.cat((img, depth), 0)
    print(rgbd.shape)

    # print(rgbd[0][2][2])  # tensor(0.8431)
    # print(rgbd[1][2][2])  # tensor(0.9020)
    # print(rgbd[2][2][2])  # tensor(0.8902)
    # print(rgbd[3][2][2])  # tensor(0.2543)












    # parse arguments
    # args = parse_args()
    # camera_files = ['camera_A.h264', 'camera_B.h264', 'camera_C.h264', 'camera_D.h264']
    # videos_path = os.path.join(args.dataset_path, 'imagery', 'side_angles')
    # video_directories = glob(videos_path + '/*/')
    # for directory in tqdm(video_directories):
        # for camera_file in camera_files:
            # file_path = os.path.join(directory, camera_file)
            # frame_dir_path = os.path.join(directory, camera_file.split('.')[0])
            # if os.path.isdir(frame_dir_path):
                # continue
            # cap = cv2.VideoCapture(file_path)
            # os.makedirs(frame_dir_path, exist_ok=True)
            # frame = None
            # count = 1
            # while cap.isOpened():
                # ret, frame = cap.read()
                # if frame is None:
                    # break
                # else:
                    # frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # image = Image.fromarray(frame)
                    # image.save(os.path.join(frame_dir_path, str(count)) + '.jpg')
                    # count += 1
