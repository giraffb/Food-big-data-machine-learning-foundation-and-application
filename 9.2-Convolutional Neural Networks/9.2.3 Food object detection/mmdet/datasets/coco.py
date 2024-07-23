# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    # METAINFO = {
    #     'classes':
    #         ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    #          'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    #          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    #          'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    #          'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    #          'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #          'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    #          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    #          'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    #          'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    #          'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #          'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    #          'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    #     # palette is a list of color tuples, which is used for visualization.
    #     'palette':
    #         [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    #          (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    #          (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
    #          (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
    #          (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
    #          (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
    #          (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
    #          (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
    #          (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
    #          (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
    #          (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
    #          (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
    #          (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
    #          (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
    #          (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
    #          (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
    #          (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
    #          (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
    #          (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
    #          (246, 0, 122), (191, 162, 208)]
    # }
    #ZSFooD
    METAINFO = {
        'classes': ('chinese sauerkraut', 'fried rice', 'egg & vegetable soup', 'sweet and sour pork tenderloin', 'breakfast rolls',
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

    #UECFOOD256
    # METAINFO ={'classes': ('rice', 'eels on rice', 'pilaf', "chicken-'n'-egg on rice", 'pork cutlet on rice', 'beef curry',
    #              'sushi', 'chicken rice', 'fried rice', 'tempura bowl', 'bibimbap', 'toast', 'croissant',
    #              'roll bread', 'raisin bread', 'chip butty', 'hamburger', 'pizza', 'sandwiches', 'udon noodle',
    #              'tempura udon', 'soba noodle', 'ramen noodle', 'beef noodle', 'tensin noodle', 'fried noodle',
    #              'spaghetti', 'Japanese-style pancake', 'takoyaki', 'gratin', 'sauteed vegetables', 'croquette',
    #              'grilled eggplant', 'sauteed spinach', 'vegetable tempura', 'miso soup', 'potage', 'sausage',
    #              'oden', 'omelet', 'ganmodoki', 'jiaozi', 'stew', 'teriyaki grilled fish', 'fried fish',
    #              'grilled salmon', 'salmon meuniere', 'sashimi', 'grilled pacific saury', 'sukiyaki',
    #              'sweet and sour pork', 'lightly roasted fish', 'steamed egg hotchpotch', 'tempura',
    #              'fried chicken', 'sirloin cutlet', 'nanbanzuke', 'boiled fish', 'seasoned beef with potatoes',
    #              'hambarg steak', 'steak', 'dried fish', 'ginger pork saute', 'spicy chili-flavored tofu',
    #              'yakitori', 'cabbage roll', 'omelet', 'egg sunny-side up', 'natto', 'cold tofu', 'egg roll',
    #              'chilled noodle', 'stir-fried beef and peppers', 'simmered pork', 'boiled chicken and vegetables',
    #              'sashimi bowl', 'sushi bowl', 'fish-shaped pancake with bean jam', 'shrimp with chill source',
    #              'roast chicken', 'steamed meat dumpling', 'omelet with fried rice', 'cutlet curry',
    #              'spaghetti meat sauce', 'fried shrimp', 'potato salad', 'green salad', 'macaroni salad',
    #              'Japanese tofu and vegetable chowder', 'pork miso soup', 'chinese soup', 'beef bowl',
    #              'kinpira-style sauteed burdock', 'rice ball', 'pizza toast', 'dipping noodles', 'hot dog',
    #              'french fries', 'mixed rice', 'goya chanpuru', 'green curry', 'okinawa soba', 'mango pudding',
    #              'almond jelly', 'jjigae', 'dak galbi', 'dry curry', 'kamameshi', 'rice vermicelli', 'paella',
    #              'tanmen', 'kushikatu', 'yellow curry', 'pancake', 'champon', 'crape', 'tiramisu', 'waffle',
    #              'rare cheese cake', 'shortcake', 'chop suey', 'twice cooked pork', 'mushroom risotto', 'samul',
    #              'zoni', 'french toast', 'fine white noodles', 'minestrone', 'pot au feu', 'chicken nugget',
    #              'namero', 'french bread', 'rice gruel', 'broiled eel bowl', 'clear soup', 'yudofu', 'mozuku',
    #              'inarizushi', 'pork loin cutlet', 'pork fillet cutlet', 'chicken cutlet', 'ham cutlet',
    #              'minced meat cutlet', 'thinly sliced raw horsemeat', 'bagel', 'scone', 'tortilla', 'tacos',
    #              'nachos', 'meat loaf', 'scrambled egg', 'rice gratin', 'lasagna', 'Caesar salad', 'oatmeal',
    #              'fried pork dumplings served in soup', 'oshiruko', 'muffin', 'popcorn', 'cream puff', 'doughnut',
    #              'apple pie', 'parfait', 'fried pork in scoop', 'lamb kebabs',
    #              'dish consisting of stir-fried potato, eggplant and green pepper', 'roast duck', 'hot pot',
    #              'pork belly', 'xiao long bao', 'moon cake', 'custard tart', 'beef noodle soup', 'pork cutlet',
    #              'minced pork rice', 'fish ball soup', 'oyster omelette', 'glutinous oil rice', 'trunip pudding',
    #              'stinky tofu', 'lemon fig jelly', 'khao soi', 'Sour prawn soup', 'Thai papaya salad',
    #              'boned, sliced Hainan-style chicken with marinated rice',
    #              'hot and sour, fish and vegetable ragout', 'stir-fried mixed vegetables', 'beef in oyster sauce',
    #              'pork satay', 'spicy chicken salad', 'noodles with fish curry', 'Pork Sticky Noodles',
    #              'Pork with lemon', 'stewed pork leg', 'charcoal-boiled pork neck', 'fried mussel pancakes',
    #              'Deep Fried Chicken Wing', 'Barbecued red pork in sauce with rice', 'Rice with roast duck',
    #              'Rice crispy pork', 'Wonton soup', 'Chicken Rice Curry With Coconut', 'Crispy Noodles',
    #              'Egg Noodle In Chicken Yellow Curry', 'coconut milk soup', 'pho', 'Hue beef rice vermicelli soup',
    #              'Vermicelli noodles with snails', 'Fried spring rolls', 'Steamed rice roll', 'Shrimp patties',
    #              'ball shaped bun with pork', 'Coconut milk-flavored crepes with shrimp and beef',
    #              'Small steamed savory rice pancake', 'Glutinous Rice Balls', 'loco moco', 'haupia', 'malasada',
    #              'laulau', 'spam musubi', 'oxtail soup', 'adobo', 'lumpia', 'brownie', 'churro', 'jambalaya',
    #              'nasi goreng', 'ayam goreng', 'ayam bakar', 'bubur ayam', 'gulai', 'laksa', 'mie ayam',
    #              'mie goreng', 'nasi campur', 'nasi padang', 'nasi uduk', 'babi guling', 'kaya toast',
    #              'bak kut teh', 'curry puff', 'chow mein', 'zha jiang mian', 'kung pao chicken', 'crullers',
    #              'eggplant with garlic sauce', 'three cup chicken', 'bean curd family style',
    #              'salt & pepper fried shrimp with shell', 'baked salmon',
    #              'braised pork meat ball with napa cabbage', 'winter melon soup', 'steamed spareribs',
    #              'chinese pumpkin pie', 'eight treasure rice', 'hot & sour soup'),
    #            'palette':[(57, 12, 140), (125, 114, 71), (52, 44, 216), (16, 15, 47), (111, 119, 13), (101, 214, 112), (229, 142, 3), (81, 216, 174), (142, 79, 110), (172, 52, 47), (194, 49, 183), (176, 135, 22), (235, 63, 193), (40, 150, 185), (98, 35, 23), (116, 148, 40), (119, 51, 194), (142, 232, 186), (83, 189, 181), (107, 136, 36), (87, 125, 83), (236, 194, 138), (112, 166, 28), (117, 16, 161), (205, 137, 33), (108, 161, 108), (255, 202, 234), (73, 135, 71), (126, 134, 219), (204, 185, 112), (70, 252, 46), (24, 56, 78), (81, 216, 32), (197, 195, 239), (128, 5, 58), (136, 174, 57), (150, 222, 80), (232, 1, 134), (91, 54, 152), (101, 78, 191), (82, 0, 165), (250, 9, 57), (185, 157, 122), (29, 123, 40), (43, 248, 35), (64, 65, 243), (84, 135, 216), (108, 102, 159), (204, 191, 224), (231, 61, 126), (115, 32, 173), (10, 117, 112), (3, 36, 30), (117, 34, 16), (169, 36, 121), (142, 248, 109), (67, 242, 124), (242, 208, 97), (48, 49, 220), (181, 216, 210), (239, 27, 50), (31, 206, 173), (55, 127, 98), (97, 229, 71), (216, 93, 142), (236, 127, 38), (226, 50, 25), (7, 47, 121), (85, 208, 248), (246, 109, 205), (30, 84, 194), (1, 199, 135), (232, 146, 216), (249, 79, 97), (151, 111, 29), (31, 160, 29), (25, 244, 80), (29, 41, 95), (35, 34, 120), (206, 61, 126), (20, 41, 214), (161, 133, 104), (160, 122, 135), (202, 67, 153), (234, 161, 37), (4, 234, 51), (37, 109, 135), (67, 178, 35), (125, 189, 145), (80, 224, 154), (4, 153, 53), (68, 135, 59), (54, 79, 139), (144, 107, 175), (104, 135, 250), (128, 26, 47), (216, 141, 22), (1, 170, 66), (134, 82, 226), (218, 4, 57), (38, 76, 18), (189, 75, 220), (65, 21, 157), (186, 20, 183), (107, 127, 52), (181, 208, 79), (121, 83, 90), (211, 12, 91), (170, 210, 127), (136, 81, 55), (195, 19, 240), (113, 102, 235), (179, 156, 116), (114, 12, 98), (204, 168, 142), (35, 142, 179), (204, 169, 14), (59, 133, 91), (135, 19, 55), (222, 176, 160), (223, 59, 197), (97, 130, 22), (223, 0, 100), (186, 220, 35), (169, 160, 63), (153, 158, 209), (167, 206, 151), (65, 98, 215), (194, 89, 154), (207, 0, 155), (146, 107, 220), (164, 238, 226), (226, 109, 242), (86, 43, 145), (171, 47, 120), (158, 115, 101), (75, 12, 23), (125, 243, 37), (233, 212, 99), (196, 253, 204), (124, 75, 2), (54, 217, 112), (90, 237, 25), (127, 62, 233), (68, 237, 162), (226, 218, 228), (81, 243, 230), (132, 126, 141), (248, 122, 140), (225, 39, 146), (120, 139, 171), (163, 41, 70), (77, 118, 196), (78, 109, 32), (212, 208, 169), (238, 212, 31), (105, 215, 199), (10, 194, 244), (3, 180, 152), (199, 214, 112), (249, 112, 139), (223, 248, 14), (199, 172, 207), (84, 239, 65), (13, 201, 13), (42, 219, 69), (236, 93, 25), (133, 194, 167), (108, 232, 167), (172, 194, 142), (215, 129, 41), (240, 9, 26), (179, 114, 35), (20, 15, 126), (102, 10, 78), (122, 64, 242), (58, 111, 238), (131, 188, 85), (58, 83, 159), (55, 13, 159), (192, 203, 101), (38, 124, 52), (154, 61, 21), (177, 219, 189), (35, 174, 6), (215, 250, 54), (221, 185, 235), (78, 222, 90), (138, 247, 238), (223, 137, 165), (125, 44, 142), (230, 124, 237), (194, 172, 14), (253, 166, 93), (249, 108, 181), (132, 174, 143), (141, 5, 97), (43, 123, 208), (250, 123, 243), (251, 229, 8), (47, 150, 113), (207, 124, 156), (188, 242, 176), (217, 169, 180), (232, 138, 156), (128, 118, 61), (98, 161, 61), (94, 98, 110), (247, 141, 144), (51, 99, 151), (116, 184, 91), (154, 7, 64), (140, 23, 27), (149, 64, 251), (52, 6, 145), (240, 245, 225), (174, 94, 26), (129, 244, 58), (33, 205, 251), (37, 27, 77), (76, 155, 43), (127, 60, 213), (115, 194, 230), (226, 152, 219), (156, 30, 50), (106, 108, 135), (41, 80, 122), (88, 38, 80), (1, 209, 230), (240, 149, 16), (118, 147, 144), (232, 36, 119), (135, 101, 217), (58, 115, 76), (136, 72, 36), (30, 84, 157), (147, 224, 63), (239, 155, 206), (139, 252, 224), (41, 20, 221), (165, 128, 13), (46, 117, 10), (137, 20, 89), (240, 226, 142), (92, 223, 251), (46, 240, 178), (209, 170, 164), (53, 82, 168), (210, 253, 147), (205, 18, 232), (45, 161, 129), (165, 59, 206), (0, 236, 211), (27, 96, 185), (255, 226, 26), (104, 136, 67)]
    #         }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                    raw_ann_info,
                'raw_img_info':
                    raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list


    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        #ZSFOOD
        data_info['img_path_d'] = img_path.split('/')[-2] + '_enhance/'\
                                  + img_path.split('/')[-1].replace('jpg', 'png')
        # data_info['img_path_d'] = img_path.replace('UECFOOD256', 'UECFOOD256_depth').replace('jpg', 'png')
        # print(data_info['img_path_d'])

        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                print(i)
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
