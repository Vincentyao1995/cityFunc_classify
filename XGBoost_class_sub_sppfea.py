#------use for class the every sppfea of subimg of hd_parcel------
import os
import numpy as np
from sklearn.externals import joblib
import numpy as np
import config
import os
import scipy
import pandas
import xgboost as xgb
import config
import os
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import datetime

hd_parcel_class2label={
    'RES':0,
    'EDU':1,
    'TRA':2,
    'GRE':3,
    'COM':4,
    'OTH':5
}

def class_one_img_folder(img_folder_path):
    pass


def get_folder_files_count(folder_path):
    count = 0
    for item in os.listdir(folder_path):
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


def generate_sppfea_sub_class(thu_sub_sppfearoot, result_folder_path):
    model = joblib.load("xgboost_thu_spp.m")
    for img_index in range(1000, 1064):
        print(img_index)
        img_folder_name = "hd_" + str(img_index)

        img_folder_path = os.path.join(thu_sub_sppfea_root, img_folder_name)
        img_sub_class_result_path = os.path.join(result_folder_path, img_folder_name + ".txt")

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
    with open(hd_parcel_label_path,'r') as f:
        line = f.readline()
        line = f.readline().strip()
        while line:
            type = hd_parcel_class2label[line.split(',')[-1]]
            index = line.split(',')[0]
            label.append(type)
            line = f.readline().strip()
    return label

if __name__ == "__main__":
    thu_sub_sppfea_root = r"/media/jsl/675e5f7b-9f40-40ed-8ef0-adf4a2832461/data/hd_parcel/hd_pacel_ss_sub_sppfea_150"
    result_folder_path = r"/media/jsl/675e5f7b-9f40-40ed-8ef0-adf4a2832461/data/hd_parcel/hd_parcel_sppfea_result"

    hd_parcel_class2label_path = r"/media/jsl/675e5f7b-9f40-40ed-8ef0-adf4a2832461/data/hd_parcel/hd_parcel_spp_fea/hd_clip_parcel_type.txt"
    lable_all = get_hd_parcel_label(hd_parcel_class2label_path)

    data = []
    label = []
    for file in os.listdir(result_folder_path):
        index = file.split('.')[0].split("_")[1]
        index = int(index)
        file_path = os.path.join(result_folder_path,file)
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
