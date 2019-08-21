import os
import os.path
import config
import numpy as np
import tensorflow as tf
import VGG
import utils
import time
import math
import os

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import datetime
# %%

def timer(func):
    def inner(*args, **kwargs):

        start = time.clock()
        func(*args, **kwargs)
        end = time.clock()
        print("time using: %s" % str(end - start))
    return inner

def classify_imgFea_from_single_img(image_path, dataset_choice = 'cityFunc'):
    '''
        This function is to test chaoyang single img classification. mainly to extract img features from single pic using vgg16+spp+softmax model, results write in .npy files, check '../dataset/haidian_streetblock/hd_clip_jpg_SPPfeature' 

        To change model, change: config.thu_checkpoint_path = r'../dataset/THUDataset/logs/train/model.ckpt-14999'

        '''
    dataset_choice = 'cityFunc'
    dataset = config.dataset_config[dataset_choice]


    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)

    img2 = tf.image.resize_nearest_neighbor([img], [dataset['img_w'], dataset['img_h']])

    x = tf.placeholder(tf.float32, shape=[1, dataset['img_w'], dataset['img_h'], 3])
    y_ = tf.placeholder(tf.int16, shape=[1, dataset['n_classes']])

    logits = VGG.VGG16N(x, dataset['n_classes'], is_pretrain=False)
    feature = tf.nn.softmax(logits)
    class_res = tf.nn.argmax(feature, 1)

    with tf.Session() as sess:
        # restore Graph
        # restore paras
        saver = tf.train.Saver()
        saver.restore(sess, dataset['checkpoint_path'])
        print("Loading trained model successfully!! ")

        img_content = sess.run(img2, feed_dict={img_path: image_path})
        class_res = sess.run(class_res, feed_dict={x: img_content})

    return class_res



def extract_imgFea_from_street_block(dataset_choice = 'cityFunc'):
    '''
    This function is main function for this .py. mainly to extract img features from street blocks using vgg16+spp+softmax model, results write in .npy files, check '../dataset/haidian_streetblock/hd_clip_jpg_SPPfeature' 
    
    To change model, change: config.thu_checkpoint_path = r'../dataset/THUDataset/logs/train/model.ckpt-14999'

    '''
    dataset = config.dataset_config[dataset_choice]
    dataset.setdefault('path_streetBlock', r'../dataset/haidian_streetblock/hd_clip_jpg')
    dataset.setdefault('path_streetBlock_SPPfeature', r'../dataset/haidian_streetblock/hd_cityFuncSPPfeature')

    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels = 3)

    img2 = tf.image.resize_nearest_neighbor([img], [dataset['img_w'], dataset['img_h']])

    x = tf.placeholder(tf.float32, shape = [1, dataset['img_w'], dataset['img_h'],3])
    y_ = tf.placeholder(tf.int16, shape = [1, dataset['n_classes']])

    logits = VGG.VGG16N(x, dataset['n_classes'], is_pretrain = False)
    feature = tf.nn.softmax(logits)
    feature = logits

    with tf.Session() as sess:
        # restore Graph
        # restore paras
        saver = tf.train.Saver()
        saver.restore(sess, dataset['checkpoint_path'])
        print("Loading trained model successfully!! ")

        streetblock_data_path = dataset['path_streetBlock']
        streetblock_feature_path = dataset['path_streetBlock_SPPfeature']

        if not os.path.exists(streetblock_feature_path):
            os.mkdir(streetblock_feature_path)

        array_img_features = np.zeros((len(os.listdir(streetblock_data_path)), dataset['n_classes']))

        for img_name in os.listdir(streetblock_data_path):

            parcel_index = int(img_name.split('.')[0].split('_')[-1])

            streetblock_path = os.path.join(streetblock_data_path, img_name)

            img_content = sess.run(img2, feed_dict = {img_path: streetblock_path})
            fea = sess.run(feature, feed_dict = {x:img_content})
            feature_res_path = os.path.join(streetblock_feature_path, img_name.split('.')[0] + '.npy')

            img_feature = np.array(fea[0])

            # save img features seperately.
            np.save(feature_res_path, img_feature)
            array_img_features[parcel_index][:] = img_feature

            res2 = [round(i,6) for i in list(fea[0])]
            print(streetblock_path, res2)

        #save the total img features in one file
        features_res_path = os.path.join(streetblock_feature_path, 'parcel_imgFea_CFSplit32B_logits')
        np.save(features_res_path, np.array(array_img_features))
        print(array_img_features)
    return array_img_features

@timer
def extract_imgFea_from_sub_region(subRegion_path, subRegion_spp_path):

    dataset = config.dataset_config['thu']
    dataset.setdefault('path_sub_region', subRegion_path)
    dataset.setdefault('path_subRegion_SPPfeature', subRegion_spp_path)

    if not os.path.exists(subRegion_spp_path):
        os.mkdir(subRegion_spp_path)

    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)
    img2 = tf.image.resize_nearest_neighbor([img], [dataset['img_w'], dataset['img_h']])

    x = tf.placeholder(tf.float32, shape=[1, dataset['img_w'], dataset['img_h'], 3])

    logits = VGG.VGG16N(x, dataset['n_classes'], is_pretrain = False)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(dataset['checkpoint_path'])
    print(ckpt)


    with tf.Session() as sess:

        saver.restore(sess, dataset['checkpoint_path'])
        print("Loading trained model successfully!! ")

        thu_folder_path = dataset['path_sub_region']
        thu_sppfea_path = dataset['path_subRegion_SPPfeature']
        if not os.path.exists(thu_sppfea_path):
            os.mkdir(thu_sppfea_path)

        for class_name in os.listdir(thu_folder_path):
            # if not class_name == "hd_0072":
            #     continue

            print("spp features compute start", class_name)
            image_class_folder = os.path.join(thu_folder_path, class_name)
            sppfea_class_folder = os.path.join(thu_sppfea_path, class_name)
            if not os.path.exists(sppfea_class_folder):
                os.mkdir(sppfea_class_folder)
            for img_name in os.listdir(image_class_folder):
                suffix = img_name[-3:]
                if not suffix == "jpg":
                    continue
                hd_parcel_img_name = os.path.join(image_class_folder, img_name)
                print(hd_parcel_img_name)
                # print(hd_parcel_img_name)
                mm2 = sess.run(img2, feed_dict={img_path: hd_parcel_img_name})
                pre = sess.run(logits, feed_dict={x: [mm2[0]]})
                img_fea = pre[0]
                img_shape = pre.shape
                height = img_shape[1]
                width = img_shape[2]
                channels = img_shape[3]
                factors = [1, 2, 3, 4]
                results = []
                for factor in factors:
                    result = np.zeros((factor, factor, channels))

                    split_heigth = math.ceil(height / factor)
                    split_width = math.ceil(width / factor)
                    for n in range(channels):
                        img_one_channel = img_fea[:, :, n]
                        # print(img_one_channel)
                        for i in range(factor):
                            for j in range(factor):
                                low_height = i * split_heigth
                                high_height = (i + 1) * split_heigth
                                left_width = j * split_width
                                right_width = (j + 1) * split_width

                                img_part = img_one_channel[low_height:high_height, left_width:right_width]
                                try:
                                    max_part = np.max(img_part)
                                except Exception as e:
                                    print("something wrong:")
                                    max_part = 0
                                result[i, j, n] = max_part

                    results.append(result)
                parcel_fea_path = os.path.join(sppfea_class_folder, img_name + '.npy')

                results = np.array(results)
                np.save(parcel_fea_path, results)
                print("spp features saves down", parcel_fea_path)

# this function is to read features extracted from sub regions, then vote out feature in a street block. mainly reference: XGBoost_class_sub_sppfea.py

# ------use for class the every sppfea of subimg of hd_parcel------



hd_parcel_class2label = {
    'RES': 6,
    'EDU': 1,
    'TRA': 2,
    'GRE': 3,
    'COM': 4,
    'OTH': 5
}

@timer
def generate_streetblock_feature(subRegion_path, subRegion_spp_path):
    """
    优化后的parcel简单场景统计
    :param img_folder_path:
    :return: 
    """
    iou_threshold = 0.8
    sub_imgs_folder_root = subRegion_spp_path
    sub_imgs_txt_root = subRegion_path
    features_res = []
    array_img_features = np.zeros((len(os.listdir(sub_imgs_folder_root)), 13))

    for img_name in os.listdir(sub_imgs_folder_root):
        if os.path.isfile(os.path.join(sub_imgs_folder_root, img_name)):
            continue
        print(img_name)
        parcel_index = int(img_name.split('_')[-1])

        sub_imgs_folder = os.path.join(sub_imgs_folder_root, img_name)
        sub_txt_folder = os.path.join(sub_imgs_txt_root, img_name)
        sub_ids_asc_area = os.path.join(sub_txt_folder, "area_asc_ids.txt")
        sub_iou_matrix = os.path.join(sub_txt_folder, "iou_matrix.txt")
        if not os.path.exists(sub_iou_matrix):
            continue

        for file in os.listdir(sub_imgs_folder):
            if file.endswith('.txt'):
                spp_fea_result_simple_scene_file = os.path.join(sub_imgs_folder, file)
        #spp_fea_result_simple_scene_file = os.path.join(sub_imgs_folder, img_name + ".txt")
        if not os.path.exists(spp_fea_result_simple_scene_file):
            continue
        print('processing %s' % spp_fea_result_simple_scene_file)
        simple_scenes_result = np.loadtxt(spp_fea_result_simple_scene_file).astype("int")
        # print(simple_scenes_result)
        fea = np.zeros(13)
        # 如果只有一个简单场景
        if not simple_scenes_result.shape:
            fea[int(simple_scenes_result)] += 1
        else:
            ids_asc_area = np.loadtxt(sub_ids_asc_area).astype("int")
            ids_asc_area = list(ids_asc_area)
            iou_matrix = np.loadtxt(sub_iou_matrix)
            n_ids = len(ids_asc_area)
            while len(ids_asc_area):
                id_base = ids_asc_area[0]
                each_type_count = np.zeros(13)
                each_type_count[simple_scenes_result[id_base]] += 1
                for id_compare in range(0, n_ids):
                    iou = iou_matrix[id_base][id_compare]
                    if iou > iou_threshold:
                        each_type_count[simple_scenes_result[id_compare]] += 1
                        if id_compare in ids_asc_area:
                            ids_asc_area.remove(id_compare)
                max_types = np.where(each_type_count)
                # print(max_types)
                for max_type in max_types:
                    max_type = max_type[0]
                    # print("max:", max_type)
                    fea[max_type] += 1

                ids_asc_area.remove(id_base)
        res_street_block_fea = os.path.join(sub_imgs_folder, "street_block_img_feature")

        array_img_features[parcel_index - 1][:] = fea # for chaoyang, there should be - 1, others, normal without - 1
        np.save(res_street_block_fea, fea)
    feas_res_path = os.path.join(sub_imgs_folder_root, 'feas_result')
    np.save(feas_res_path, array_img_features)
    print(array_img_features, array_img_features.shape)

def class_one_img_folder(img_folder_path):
    pass

def get_folder_files_count(folder_path):
    count = 0
    for item in os.listdir(folder_path):
        if item.split('.')[-1] == 'npy':
            count += 1
    return count

def test_file_number(thu_sub_sppfea_root):
    for img_name in os.listdir(thu_sub_sppfea_root):
        img_folder_path = os.path.join(thu_sub_sppfea_root, img_name)
        count = 0
        for item in os.listdir(img_folder_path):
            if item[:3] != "sub":
                print(img_name)

def read_one_spp_fea(sppfea_path):
    fea = np.load(sppfea_path)
    fea_reshape = np.array([])
    for f in fea:
        f2 = f.reshape(-1)
        fea_reshape = np.append(fea_reshape, f2)
    return fea_reshape

@timer
def classify_subRegion_sppFea(thu_sub_sppfea_root):

    model_path = "xgboost_basicScene.pkl"
    model = joblib.load(model_path)
    for img_name in os.listdir(thu_sub_sppfea_root):
        img_folder_path = os.path.join(thu_sub_sppfea_root, img_name)
        if os.path.isfile(img_folder_path):
            continue
        print('processing %s' % img_folder_path)
        img_sub_class_result_path = os.path.join(img_folder_path, img_name + ".txt")

        n = get_folder_files_count(img_folder_path)
        test_x = []
        if n == 0:
            continue
        for index in range(0, n):
            npy_file_name = "sub_" + str(index) + ".jpg.npy"
            npy_file_path = os.path.join(img_folder_path, npy_file_name)
            fea = read_one_spp_fea(npy_file_path)

            test_x.append(fea)
        test_x = np.array(test_x)
        class_label = model.predict(test_x)
        print(class_label)
        np.savetxt(img_sub_class_result_path, class_label)

def get_hd_parcel_label(hd_parcel_label_path):
    label = []
    with open(hd_parcel_label_path, 'r') as f:
        line = f.readline()
        line = f.readline().strip()
        while line:
            type = hd_parcel_class2label[line.split(',')[-1]]
            index = line.split(',')[0]
            label.append(type)
            line = f.readline().strip()
    return label

def xgboost_street_block_imgFeature():

    thu_sub_sppfea_root = r"./dataset/haidian_streetblock/hd_subRegion_SPPfeature"
    result_folder_path = r"./dataset/haidian_streetblock/hd_parcel_sppfea_result"

    hd_parcel_class2label_path = r"./dataset/hd_clip_parcel_type.txt"
    lable_all = get_hd_parcel_label(hd_parcel_class2label_path)

    data = []
    label = []
    for file in os.listdir(result_folder_path):
        index = file.split('.')[0].split("_")[1]
        index = int(index)
        file_path = os.path.join(result_folder_path, file)
        fea = np.loadtxt(file_path)
        fea2 = np.zeros(13)
        if fea.shape:
            print(index)
            for i in fea:
                i = int(i)
                fea2[i] += 1
        else:
            fea2[int(fea)] += 1

        print(index, fea2)
        data.append(fea2)
        label.append(lable_all[index])

    data = np.array(data)
    label = np.array(label)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("XGBoost start train...")
    print(start_time)
    model = XGBClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, "xgboost_parcel_spp.m")
    y_pred = model.predict(x_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("XGBoost end train...")
    print(end_time)

def vgg16_basicScene_feature_extractor():
    '''
        This function is to fit xgboost classifier using VGG16 scene feature extractor' 
    '''
    dataset = config.dataset_config['thu']

    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)

    img2 = tf.image.resize_nearest_neighbor([img], [dataset['img_w'], dataset['img_h']])

    x = tf.placeholder(tf.float32, shape=[1, dataset['img_w'], dataset['img_h'], 3])
    y_ = tf.placeholder(tf.int16, shape=[1, dataset['n_classes']])

    logits = VGG.VGG16N(x, dataset['n_classes'], is_pretrain=False)


    with tf.Session() as sess:
        # restore Graph
        # restore paras
        saver = tf.train.Saver()
        saver.restore(sess, dataset['checkpoint_path'])
        print("Loading trained model successfully!! ")

        data_root_path = os.path.join(dataset['data_path'], 'train')

        array_img_features = []
        array_img_labels = []

        for class_name in os.listdir(data_root_path):
            class_root_path = os.path.join(data_root_path, class_name)
            for img_name in os.listdir(class_root_path):

                img_root_path = os.path.join(class_root_path, img_name)
                print('processing %s ' % img_root_path)

                label_root_img = dataset['class2label'][class_name]
                array_img_labels.append(label_root_img)

                img_content = sess.run(img2, feed_dict={img_path: img_root_path})
                pre = sess.run(logits, feed_dict={x: img_content})

                img_fea = pre[0]
                img_shape = pre.shape
                height = img_shape[1]
                width = img_shape[2]
                channels = img_shape[3]
                factors = [1, 2, 3, 4]
                results = []
                for factor in factors:
                    result = np.zeros((factor, factor, channels))

                    split_heigth = math.ceil(height / factor)
                    split_width = math.ceil(width / factor)
                    for n in range(channels):
                        img_one_channel = img_fea[:, :, n]
                        # print(img_one_channel)
                        for i in range(factor):
                            for j in range(factor):
                                low_height = i * split_heigth
                                high_height = (i + 1) * split_heigth
                                left_width = j * split_width
                                right_width = (j + 1) * split_width

                                img_part = img_one_channel[low_height:high_height, left_width:right_width]
                                try:
                                    max_part = np.max(img_part)
                                except Exception as e:
                                    print("something wrong:")
                                    max_part = 0
                                result[i, j, n] = max_part

                    # print(result.shape)
                    # print(result)

                    results.append(result)

                fea_reshape = np.array([])
                for f in results:
                    f2 = f.reshape(-1)
                    fea_reshape = np.append(fea_reshape, f2)
                array_img_features.append(fea_reshape)

        # save the total img features in one file
        array_img_features = np.array(array_img_features)
        array_img_labels = np.array(array_img_labels)

        xgboost_fitting_features_path = os.path.join(dataset['data_path'],'xgboost_train_features.npy' )
        xgboost_fitting_labels_path = os.path.join(dataset['data_path'],'xgboost_train_labels.npy')
        np.save(xgboost_fitting_features_path, array_img_features)
        np.save(xgboost_fitting_labels_path, array_img_labels)


    return array_img_features, array_img_labels

def xgboost_basicScene_feature_fitting():
    feature, label = vgg16_basicScene_feature_extractor()
    model = XGBClassifier()
    model.fit(feature, label)
    train_X, test_X, train_y, test_y = train_test_split(feature, label, test_size=0.2)
    joblib.dump(model, 'xgboost_basicScene.pkl')

    y_pred = model.predict(test_X)
    acc = accuracy_score(test_y, y_pred)

    print('trianing acc: %f'% acc)


if __name__ == '__main__':

    #xgboost_basicScene_feature_fitting()

    subRegion_path =  r'../dataset/chaoyang/imgs/sub_regions'
    subRegion_spp_path = r'../dataset/chaoyang/imgs/sub_regions_spp'

    # 1) after using selective search generating sub region imgs, extract vgg16_spp1234 features from them.
    #extract_imgFea_from_sub_region(subRegion_path, subRegion_spp_path)

    # 2) classify these features.
    #classify_subRegion_sppFea(subRegion_spp_path)

    # 3) word parcel to count class, get ss feature.
    generate_streetblock_feature(subRegion_path, subRegion_spp_path)


    #extract_imgFea_from_street_block()
