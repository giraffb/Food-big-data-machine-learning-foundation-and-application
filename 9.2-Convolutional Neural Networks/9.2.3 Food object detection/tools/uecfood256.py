# import os
# import json
# from PIL import Image
# import random
# from sklearn.model_selection import train_test_split
#
# # 路径设置
# data_dir = '/home/stu_6/NLM/datasets/UECFOOD256/'
#
# # 读取类别信息
# categories = []
# mminfo = {'classes':[]}
# with open(os.path.join(data_dir, 'category.txt'), 'r') as f:
#     for line in f.readlines():
#         print(line)
#         i = 0
#         while(line[i]!='	' and i < len(line)):
#             i += 1
#
#
#         # category_name = ''
#         # category_id = ''
#
#             # if line[i]==' ':
#         category_id = line[:i]
#         category_name = line[i + 1:-1]
#
#         # category_id, category_name = line.split(' ')
#         # print(category_id)
#         categories.append({
#             'id': int(category_id),
#             'name': category_name,
#             'supercategory': 'food'
#         })
#         mminfo['classes'].append(category_name)
#
#
# # print(mminfo)
# # 读取标注信息
# images = []
# annotations = []
# annotation_id = 1
#
# for category in categories:
#     category_id = category['id']
#     category_dir = os.path.join(data_dir, str(category_id))
#     p = 0
#     with open(os.path.join(category_dir, 'bb_info.txt'), 'r') as f:
#         for line in f.readlines():
#             # print(line)
#             if p == 0:
#                 p = 1
#                 continue
#             image_file, x1, y1, x2, y2 = line.split(' ')
#             image_file = image_file + '.jpg'
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#
#             # 获取图像路径
#             image_path = os.path.join(category_dir, image_file)
#             # print(image_path)
#             if not os.path.exists(image_path):
#                 continue
#
#             # 读取图像大小
#             with Image.open(image_path) as img:
#                 width, height = img.size
#
#             # 获取图像ID
#             image_id = len(images) + 1
#
#             # 添加图像信息
#             images.append({
#                 'id': image_id,
#                 'file_name': os.path.join(str(category_id), image_file),
#                 'width': width,
#                 'height': height
#             })
#
#             # 添加标注信息
#             annotations.append({
#                 'id': annotation_id,
#                 'image_id': image_id,
#                 'category_id': category_id,
#                 'bbox': [x1, y1, x2 - x1, y2 - y1],
#                 'area': (x2 - x1) * (y2 - y1),
#                 'iscrowd': 0
#             })
#             annotation_id += 1
#
# # 将图像和标注按image_id进行分组
# image_annotations = {}
# for annotation in annotations:
#     image_id = annotation['image_id']
#     if image_id not in image_annotations:
#         image_annotations[image_id] = []
#     image_annotations[image_id].append(annotation)
#
# # 获取所有的image_id并划分训练集和测试集
# all_image_ids = [img['id'] for img in images]
# train_image_ids, test_image_ids = train_test_split(all_image_ids, test_size=0.2, random_state=42)
#
# # 根据划分结果生成训练集和测试集
# train_images = [img for img in images if img['id'] in train_image_ids]
# test_images = [img for img in images if img['id'] in test_image_ids]
#
# train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
# test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]
#
# # 生成COCO格式的训练集和测试集JSON文件
# train_coco_format = {
#     'images': train_images,
#     'annotations': train_annotations,
#     'categories': categories
# }
#
# test_coco_format = {
#     'images': test_images,
#     'annotations': test_annotations,
#     'categories': categories
# }
#
# # 保存为JSON文件
# train_output_file = '/home/stu_6/NLM/datasets/UECFOOD256/UECFOOD256_train_coco_format.json'
# test_output_file = '/home/stu_6/NLM/datasets/UECFOOD256/UECFOOD256_test_coco_format.json'
#
# with open(train_output_file, 'w') as f:
#     json.dump(train_coco_format, f)
#
# with open(test_output_file, 'w') as f:
#     json.dump(test_coco_format, f)
#
#
# import random
#
#
# def generate_palette(num_classes):
#     """
#     Generate a color palette for visualizing segmentation masks.
#
#     Parameters:
#         num_classes (int): Number of classes in the dataset.
#
#     Returns:
#         list: A list of RGB tuples representing the color palette.
#     """
#     random.seed(42)  # For reproducibility
#     palette = []
#     for i in range(num_classes):
#         r = random.randint(0, 255)
#         g = random.randint(0, 255)
#         b = random.randint(0, 255)
#         palette.append((r, g, b))
#     return palette
#
#
# num_classes = 256
# palette = generate_palette(num_classes)
# print(palette)
# # Print the generated palette
# # for i, color in enumerate(palette):
# #     print(f"Class {i + 1}: {color}")



import os
from PIL import Image

# Define the source and destination directories
source_dir = "/home/stu_6/NLM/datasets/UECFOOD256_depth/"
destination_dir = "/home/stu_6/NLM/datasets/UECFOOD256_depth_resized/"
new_size = (600, 480)

# Function to resize images
def resize_image(input_path, output_path, size):
    with Image.open(input_path) as img:
        img_resized = img.resize(size, Image.ANTIALIAS)
        img_resized.save(output_path)

# Function to process all images in the directory
def process_directory(source, destination, size):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for root, _, files in os.walk(source):
        relative_path = os.path.relpath(root, source)
        dest_dir = os.path.join(destination, relative_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(dest_dir, file)
                resize_image(source_file, destination_file, size)

# Process the directory
process_directory(source_dir, destination_dir, new_size)
