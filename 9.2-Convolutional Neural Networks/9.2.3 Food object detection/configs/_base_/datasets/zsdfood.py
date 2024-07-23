# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/stu_6/NLM/datasets/ZSFooD/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator





METAINFO = {
    'classes':('chinese sauerkraut', 'fried rice', 'egg & vegetable soup', 'sweet and sour pork tenderloin', 'breakfast rolls',
        'congee', 'zucchini and soybean soup', 'diced chicken with green pepper', 'wonton', 'celery fried meat',
        'pumpkin puree', 'pickled vegetable/kimchi', 'chinese cabbage and tofu', 'egg dumpling',
        'fried shrimp with corn', 'sauerkraut ', 'fried ham', 'fried beef with green pepper', 'roasted duck',
        'jellied bean curd', 'braised ribbonfish in brown sauce', 'meatballs', 'braised drumstick with sauce',
        'crispy cod stick', 'spaghetti with meat sauce', 'smoked fish', 'pickles squid', 'pork with green pepper',
        'grilled duck', 'fried beans with potatoes', 'stir fried shredded potato', 'pickled radish', 'ham pizza',
        'purple cabbage', 'porridge', 'mapo tofu', 'braised okra with mushrooms', 'steamed bread with brown sugar',
        'baozi stuffed with red bean paste', 'fresh fruit salad', 'fried lentils', 'brown sugar cake',
        'fried pork slices with green pepper', 'fried soybean sprouts with celery', 'mushroom pizza', 'bean curd',
        'fried pork livers', 'fried eggplant', 'lettuce fried meat', 'vinegar vermicelli',
        'stewed tofu with green pepper', 'mizone sports beverage', 'fried flammulina velutipes with cauliflower',
        'steak', 'cabbage vermicelli', 'pork bun', 'milk', 'chinese cabbage and vermicelli', 'roast duck meat',
        'egg cakes', 'seaweed and egg soup', 'chicken with pepper', 'fried chicken wings',
        'sweet and sour chicken breast', 'onions scrambled eggs', 'sour and spicy shredded potatoes',
        'dry-pot potato chips', 'roast meat with mushrooms', 'potatoes', 'egg soup', 'stir fried seasonal vegetables',
        'plum peanuts', 'noodle', 'fried mushroom slices', 'air-dried beef', 'steamed buns', 'kungpao chicken',
        'seaweed soup', 'seafood noodles', 'egg', 'tofu with preserved eggs', 'spring onion rolls',
        'boiled cauliflower', 'large meatball', 'crab paste', 'egg cake', 'fried cauliflower',
        'steamed bread with quicksand and milk', 'bean sprouts', 'home-style sauteed tofu', 'sauteed shrimps',
        'the fungus fried meat', 'saute spicy chicken', 'braised potato in brown sauce',
        'traditional chinese rice-pudding', 'sausages with potatoes', 'fried beans',
        'fried glutinous rice balls with sesame', 'millet congee', 'stewed seafood', 'fried mushroom with green beans',
        'fried green pepper with potato', 'vegetable bag', 'fried dumplings', 'marinated chicken feets', 'sausages',
        'preserved vegetable roast meat', 'fried chicken with green beans', 'sauteed bean sprouts', 'celery tofu',
        'chinese cabbage', 'sugared tomato', 'ice tea', 'pork chop', 'roast chicken', 'stir-fried noodles',
        'pickled green bean', 'taiwan rice roll', 'braised flammulina velutipes', 'peanuts',
        'stir fried soybeans with pickles', 'fried chicken drumsticks', 'french fries', 'filet steak', 'cauliflower',
        'rice wine soup', 'steamed white crab', 'banana milk cake', 'scrambled eggs with tomatoes', 'pineapple pizza',
        'shredded pork with garlic sauce', 'rice', 'marinated beef', 'fried crucian carp', 'fruit pizza',
        'fried sole fish', 'tomato soup', 'steamed egg', 'black rice porridge', 'spicy duck', 'mixed mungbean sprouts',
        'moon cake', 'pumpkin pie', 'onion rings', 'japanese tofu', 'fried zucchini', 'roast beef with onion', 'banana',
        'green vegetables and mushrooms', 'steamed whole grains', 'peanut milk', 'corn', 'fried shrimp', 'boiled eggs',
        'seaweed salad', 'roast chicken wings', 'salad', 'fried mushrooms', 'fried potato slices', 'sausage',
        'dough sticks(half)', 'seafood rice', 'shredded chicken soup', 'cold mixed vermicelli',
        'braised eggplant in brown sauce', 'braised spareribs', 'chicken blocks', 'beef jerky rice', 'stewed noodles',
        'roasted pork ribs', 'tomatoes', 'steamed chicken with chili sauce', 'orleans chicken wings', 'garlic bread ',
        'special chicken nuggets', 'steamed yellow croakers', 'classic pizza', 'beer duck', 'lotus-root slices',
        'potato beef', 'marinated cold cucumber', 'onion roasted loin', 'noodles with chicken sauce', 'fried potatoes',
        'sweet potato lele', 'grilled shrimp', 'fried chicken cutlet', 'filet mignon pasta',
        'fried cauliflower with carrot', 'zucchini', 'corn buns', "grandma's pickled mustard", 'supreme pizza',
        'coconut shake', 'cold cucumber', 'fried squid with pickled cabagges', 'fried gourd', 'fried chicken wing root',
        'cold beans', 'duck leg', 'sweet and sour pork ribs', 'chongqing style boiled blood curd', 'cold tofu',
        'pumpkin congee', 'steamed pork dumplings', 'soybean milk', 'cooked shrimp', 'braised pork in brown sauce',
        'fried okra', 'salad with mushroom', 'fried string beans', 'thousand-leaf tofu', 'noodles with scallion oil',
        'cabbage rice noodles', 'boiled fish with pickled cabbage and chili', 'steamed egg with minced meat',
        'shredded cabbage', 'gluten', 'homemade yuba', 'apple', 'red jujube pumpkin', 'egg tart', 'fried lettuce',
        'sweet and sour chicken nuggets', 'fried eggs with beans', 'popcorn chicken', 'boiled cabbage',
        'fried chicken gizzard'),

    # palette is a list of color tuples, which is used for visualization.
    'palette': [(233, 216, 11), (199, 79, 243), (1, 136, 55), (30, 112, 4), (255, 73, 172), (22, 34, 131),
                (166, 80, 107),
                (144, 172, 6), (43, 233, 28), (34, 32, 143), (13, 122, 105), (92, 85, 95), (205, 90, 227),
                (121, 103, 62),
                (250, 78, 71), (118, 202, 100), (138, 225, 248), (160, 139, 154), (248, 168, 158), (85, 240, 35),
                (244, 80, 74), (218, 153, 62), (147, 211, 98), (5, 70, 2), (243, 169, 193), (254, 125, 109),
                (192, 91, 227),
                (97, 88, 34), (67, 236, 14), (7, 73, 214), (154, 87, 102), (70, 139, 157), (238, 167, 166),
                (190, 126, 25),
                (128, 71, 196), (129, 5, 244), (147, 37, 142), (108, 205, 92), (130, 50, 210), (173, 92, 34),
                (117, 250, 58), (131, 201, 87), (125, 192, 83), (245, 255, 133), (138, 4, 133), (26, 52, 73),
                (182, 162, 73), (121, 141, 208), (191, 20, 34), (36, 69, 37), (142, 174, 139), (151, 60, 78),
                (199, 150, 224), (67, 86, 222), (137, 90, 101), (228, 252, 21), (196, 2, 61), (86, 139, 144),
                (206, 58, 106), (231, 77, 201), (168, 23, 57), (220, 91, 88), (90, 210, 56), (83, 146, 73),
                (48, 193, 100),
                (214, 85, 157), (15, 231, 75), (128, 25, 169), (86, 120, 53), (122, 229, 122), (238, 39, 41),
                (211, 86, 253), (36, 27, 153), (64, 1, 97), (200, 182, 15), (56, 158, 73), (174, 250, 44),
                (72, 210, 222),
                (240, 213, 177), (41, 132, 40), (127, 181, 91), (103, 156, 230), (125, 105, 52), (26, 190, 177),
                (102, 56, 123), (118, 225, 181), (132, 41, 123), (58, 248, 187), (116, 70, 102), (13, 211, 71),
                (168, 184, 0), (187, 191, 58), (249, 85, 248), (193, 176, 230), (96, 90, 3), (209, 152, 105),
                (212, 203, 156), (123, 29, 249), (85, 153, 110), (143, 184, 38), (121, 214, 68), (187, 188, 188),
                (129, 198, 226), (53, 149, 42), (117, 36, 204), (129, 250, 9), (208, 130, 171), (44, 113, 208),
                (241, 56, 202), (12, 188, 188), (122, 127, 228), (17, 250, 5), (129, 113, 46), (213, 27, 209),
                (61, 126, 132), (102, 145, 131), (211, 105, 227), (183, 104, 36), (175, 134, 89), (58, 230, 221),
                (52, 147, 44), (36, 224, 124), (162, 47, 67), (252, 189, 226), (142, 142, 241), (195, 107, 98),
                (192, 146, 194), (48, 41, 25), (27, 143, 206), (99, 252, 179), (78, 161, 99), (92, 40, 134),
                (8, 186, 78),
                (147, 42, 189), (128, 75, 71), (81, 112, 11), (134, 240, 100), (143, 127, 28), (120, 96, 91),
                (250, 7, 197),
                (166, 182, 75), (89, 117, 80), (156, 36, 119), (28, 93, 169), (168, 165, 83), (113, 234, 100),
                (237, 112, 139), (102, 167, 129), (215, 133, 102), (112, 46, 56), (87, 192, 62), (116, 222, 74),
                (110, 128, 203), (116, 174, 135), (159, 232, 17), (153, 234, 4), (45, 94, 154), (118, 229, 131),
                (70, 249, 255), (107, 52, 234), (55, 176, 150), (68, 40, 233), (222, 235, 14), (215, 188, 37),
                (138, 22, 183), (226, 176, 65), (68, 125, 104), (172, 227, 198), (213, 201, 138), (16, 180, 245),
                (156, 77, 119), (213, 130, 248), (82, 239, 163), (11, 217, 213), (87, 155, 89), (39, 63, 252),
                (207, 154, 84), (174, 178, 183), (66, 188, 127), (131, 39, 194), (50, 188, 97), (99, 194, 190),
                (66, 179, 1), (214, 183, 113), (43, 187, 11), (5, 208, 92), (88, 16, 225), (228, 221, 173),
                (225, 0, 216),
                (234, 112, 88), (101, 121, 64), (215, 13, 214), (219, 223, 153), (187, 199, 241), (216, 36, 98),
                (247, 153, 66), (190, 121, 101), (213, 187, 172), (194, 130, 234), (206, 124, 252), (15, 135, 231),
                (24, 39, 15), (69, 88, 195), (131, 74, 216), (198, 6, 251), (132, 159, 150), (198, 123, 135),
                (110, 164, 206), (112, 97, 214), (135, 201, 41), (41, 190, 230), (172, 159, 239), (222, 66, 54),
                (75, 38, 200), (194, 243, 84), (85, 51, 169), (44, 27, 111), (56, 150, 166), (212, 119, 123),
                (7, 119, 48),
                (245, 161, 54), (108, 138, 68), (46, 211, 229), (134, 152, 60), (44, 60, 219), (191, 58, 168),
                (89, 179, 98), (36, 53, 142)]
}


# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
