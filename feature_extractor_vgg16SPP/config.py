import os
'''def convert_format(): use cv2, too slow to download, ignore temporarily'''
#import cv2
import random

drop_out = {'switch':False, 'rate':0.5}

cityFunc_checkpoint_path = r'./dataset/cityFuncDataset/logs/train/cityFuncSpp8:2/model.ckpt-14999'
cityFunc_imgW = 256
cityFunc_imgH = 256
cityFunc_nClasses = 6
cityFunc_data_path = r'../dataset/cityFuncDataset'
#cityFunc_data_path = r'../dataset/chaoyang'


cityFunc_class = ['RES',
             'EDU',
             'TRA',
             'GRE',
             'COM',
             'OTH'
             ]

cityFunc_class2label = { 'RES': 1,
                    'EDU': 2,
                    'TRA': 3,
                    'GRE': 4,
                    'COM': 5,
                    'OTH': 6}

thu_checkpoint_path = r'../dataset/THUDataset/logs/train/model.ckpt-14999'
thu_imgW = 256
thu_imgH = 256
thu_nClasses = 13
thu_data_path = r'../dataset/THUDataset'

thu_class = ['airplane',
             'airport',
             'commercial_area',
             'green',
             'ground_track_field',
             'industrial_area',
             'lake',
             'parking_lot',
             'railway_station',
             'residential',
             'runway',
             'tennis_court',
             'unuse'
             ]
thu_class2label = {'airplane':0,
             'airport':1,
             'commercial_area':2,
             'green':3,
             'ground_track_field':4,
             'industrial_area':5,
             'lake':6,
             'parking_lot':7,
             'railway_station':8,
             'residential':9,
             'runway':10,
             'tennis_court':11,
             'unuse':12
             }

ucm_checkpoint_path = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/logs/ucm_spp_rotated/train/model.ckpt-14999'
ucm_checkpoint_path = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/logs/ucm_rotated/train/model.ckpt-14999'
ucm_checkpoint_path = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/logs/train/model.ckpt-10000'
ucm_imgW = 256
ucm_imgH = 256
ucm_nClasses = 21
ucm_data_path = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM'

ucm_class = ['agricultural',
             'airplane',
             'baseballdiamond',
             'beach',
             'buildings',
             'chaparral',
             'denseresidential',
             'forest',
             'freeway',
             'golfcourse',
             'harbor',
             'intersection',
             'mediumresidential',
             'mobilehomepark',
             'overpass',
             'parkinglot',
             'river',
             'runway',
             'sparseresidential',
             'storagetanks',
             'tenniscourt',
             ]
ucm_class2label = {'agricultural': 0,
               'airplane': 1,
               'baseballdiamond': 2,
               'beach': 3,
               'buildings': 4,
               'chaparral': 5,
               'denseresidential': 6,
               'forest': 7,
               'freeway': 8,
               'golfcourse': 9,
               'harbor': 10,
               'intersection': 11,
               'mediumresidential': 12,
               'mobilehomepark': 13,
               'overpass': 14,
               'parkinglot': 15,
               'river': 16,
               'runway': 17,
               'sparseresidential': 18,
               'storagetanks': 19,
               'tenniscourt': 20,
               }


# ------------ NWPU config------------
nwpu_class = ['airplane',
                    'airport',
                    'baseball_diamond',
                    'basketball_court',
                    'beach',
                    'bridge',
                    'chaparral',
                    'church',
                    'circular_farmland',
                    'cloud',
                    'commercial_area',
                    'dense_residential',
                    'desert',
                    'forest',
                    'freeway',
                    'golf_course',
                    'ground_track_field',
                    'harbor',
                    'industrial_area',
                    'intersection',
                    'island',
                    'lake',
                    'meadow',
                    'medium_residential',
                    'mobile_home_park',
                    'mountain',
                    'overpass',
                    'palace',
                    'parking_lot',
                    'railway',
                    'railway_station',
                    'rectangular_farmland',
                    'river',
                    'roundabout',
                    'runway',
                    'sea_ice',
                    'ship',
                    'snowberg',
                    'sparse_residential',
                    'stadium',
                    'storage_tank',
                    'tennis_court',
                    'terrace',
                    'thermal_power_station',
                    'wetland']

nwpu_class2label = {'airplane': 0,
                    'airport': 1,
                    'baseball_diamond': 2,
                    'basketball_court': 3,
                    'beach': 4,
                    'bridge': 5,
                    'chaparral': 6,
                    'church': 7,
                    'circular_farmland': 8,
                    'cloud': 9,
                    'commercial_area': 10,
                    'dense_residential': 11,
                    'desert': 12,
                    'forest': 13,
                    'freeway': 14,
                    'golf_course': 15,
                    'ground_track_field': 16,
                    'harbor': 17,
                    'industrial_area': 18,
                    'intersection': 19,
                    'island': 20,
                    'lake': 21,
                    'meadow': 22,
                    'medium_residential': 23,
                    'mobile_home_park': 24,
                    'mountain': 25,
                    'overpass': 26,
                    'palace': 27,
                    'parking_lot': 28,
                    'railway': 29,
                    'railway_station': 30,
                    'rectangular_farmland': 31,
                    'river': 32,
                    'roundabout': 33,
                    'runway': 34,
                    'sea_ice': 35,
                    'ship': 36,
                    'snowberg': 37,
                    'sparse_residential': 38,
                    'stadium': 39,
                    'storage_tank': 40,
                    'tennis_court': 41,
                    'terrace': 42,
                    'thermal_power_station': 43,
                    'wetland': 44}

nwpu_checkpoint_path = r'/home/vincent/Desktop/research/vin/spp_vgg/data/NWPU/logs/VGG16/train/model.ckpt-14999'
nwpu_checkpoint_path = r'/home/vincent/Desktop/research/vin/spp_vgg/data/NWPU/logs/VGG16_spp/train/model.ckpt-14999'

nwpu_imgW = 224
nwpu_imgH = 224
nwpu_nClasses = 45
nwpu_data_path = r'/home/vincent/Desktop/jsl_thesis/grad thesis/data/NWPU/NWPU-RESISC45_224'


dataset_config = {
    'nwpu':
        {'checkpoint_path': nwpu_checkpoint_path,
         'n_classes': nwpu_nClasses,
         'data_path': nwpu_data_path,
         'class2label': nwpu_class2label,
         'img_h': nwpu_imgH,
         'img_w': nwpu_imgW,
         'class': nwpu_class,
         'out_fig_name': r'nwpu_cf.png'
         },
    'ucm':
        {'checkpoint_path': ucm_checkpoint_path,
         'n_classes': ucm_nClasses,
         'data_path': ucm_data_path,
         'class2label': ucm_class2label,
         'img_h': ucm_imgH,
         'img_w': ucm_imgW,
         'class': ucm_class,
         'out_fig_name': r'ucm_cf.png'
         },
     'thu':
        {'checkpoint_path': thu_checkpoint_path,
         'n_classes': thu_nClasses,
         'data_path': thu_data_path,
         'class2label': thu_class2label,
         'img_h': thu_imgH,
         'img_w': thu_imgW,
         'class': thu_class,
         'out_fig_name': r'thu_cf.png'
         },
    'cityFunc':
        {'checkpoint_path': cityFunc_checkpoint_path,
         'n_classes': cityFunc_nClasses,
         'data_path': cityFunc_data_path,
         'class2label': cityFunc_class2label,
         'img_h': cityFunc_imgH,
         'img_w': cityFunc_imgW,
         'class': cityFunc_class,
         'out_fig_name': r'cityFunc_cf.png'
         }
}

current_dataset = dataset_config['ucm']

def get_class_name_by_index(index):
    for item in class2label:
        if class2label[item] == index:
            return item

def delete_add_img(jpg_img_path):
    for type in os.listdir(jpg_img_path):
        type_path = jpg_img_path + os.sep + type
        for class_floder in os.listdir(type_path):
            class_floder_path = type_path + os.sep + class_floder
            print(class_floder_path)
            for img in os.listdir(class_floder_path):
                img_path = class_floder_path + os.sep + img
                print(img_path)
                if img.find('_') != -1:
                    os.remove(img_path)
                    print(img)


def convert_format_jpg(jpg_img_path):
    new_jpg_rootpath = jpg_img_path.replace('bak', 'new')
    print(new_jpg_rootpath)
    if not os.path.exists(new_jpg_rootpath):
        os.mkdir(new_jpg_rootpath)
    for type in os.listdir(jpg_img_path):
        type_path = jpg_img_path + os.sep + type
        print(type_path)
        new_type_path = type_path.replace('bak', 'new')
        print(new_type_path)
        if not os.path.exists(new_type_path):
            os.mkdir(new_type_path)
        for class_floder in os.listdir(type_path):
            class_floder_path = type_path + os.sep + class_floder
            new_class_folder_path = class_floder_path.replace('bak', 'new')
            if not os.path.exists(new_class_folder_path):
                os.mkdir(new_class_folder_path)
            for img in os.listdir(class_floder_path):
                img_path = class_floder_path + os.sep + img
                new_img_path = img_path.replace('bak', 'new')
                # img_content = cv2.imread(img_path)
                # cv2.imwrite(new_img_path, img_content)


if __name__ == '__main__':
    # convert_tif2jpg(tif_folder_path, jpg_folder_path)
    # split_train_test()
    # jpg_path = r'/media/jsl/ubuntu/UCMerced_LandUse/jpgImages_bak'
    # delete_add_img(jpg_path)
    # convert_format_jpg(jpg_path)
    a = get_class_name_by_index(1)
    print(a)