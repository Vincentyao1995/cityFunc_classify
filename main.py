#from feature_extractor_vgg16SPP import fea_extractor
from process_poi import statistic_parcel_poi
import numpy as np
import classifier_xgboost_feature as cls

def process_mobike():
    # this function is to transform 24hour mobike data to one 3hour-1period data, and then concate them
    fea1 = np.loadtxt(r'./dataset/parcel_start_weekend_norm.txt')
    fea2 = np.loadtxt(r'./dataset/parcel_end_weekend_norm.txt')
    fea3 = np.loadtxt(r'./dataset/parcel_start_week_norm.txt')
    fea4 = np.loadtxt(r'./dataset/parcel_end_week_norm.txt')

    feas = [fea1,fea2, fea3, fea4]

    new_fea1 = np.zeros((fea1.shape[0], int(fea1.shape[1]/3)))
    new_fea2 = np.zeros((fea1.shape[0], int(fea1.shape[1]/3)))
    new_fea3 = np.zeros((fea1.shape[0], int(fea1.shape[1]/3)))
    new_fea4 = np.zeros((fea1.shape[0], int(fea1.shape[1]/3)))

    new_feas = [new_fea1, new_fea2, new_fea3, new_fea4]

    res = []

    for i in range(len(new_feas)):
        fea = feas[i]
        new_fea = new_feas[i]

        new_value = 0.0
        # fea1.shape[0] == 1064, fea1.shape[1] == 24
        for index_parcel in range(fea1.shape[0]):
            for index_hour in range(fea1.shape[1]):
                if (index_hour) % 3 == 0 and index_hour != 0:
                    index_period = int(index_hour / 3) - 1
                    new_fea[index_parcel][index_period] = new_value
                    new_value = 0.0
                if index_hour + 1 == fea1.shape[1]:
                    index_period = int((index_hour+1) / 3) - 1
                    new_fea[index_parcel][index_period] = new_value
                new_value += fea[index_parcel][index_hour]

    parcel_num = new_feas[0].shape[0] #1064
    period_num = new_feas[0].shape[1] #8
    res_concate = np.zeros((parcel_num, period_num*4))
    for i in range(len(new_feas)):
        fea = new_feas[i]
        for index_parcel in range(parcel_num):
            res_concate[index_parcel][i*period_num:(i+1)*period_num] = fea[index_parcel][:]

    parcel_mobike_path = r'./dataset/parcel_mobike.txt'
    np.savetxt(parcel_mobike_path, res_concate)

    return res_concate

def norm_feature(feas):
    n = feas.shape[0]

    for i in range(n):
        feas[i] = (feas[i]-min(feas[i]))/(max(feas[i])-min(feas[i])+0.0000001)
    feas = np.nan_to_num(feas)
    return feas



if __name__ == '__main__':
    #all three parcel_features.txt is writen according to parcel index strictly.row num in .txt is parcel number.

    #cal and save street block image features
    #img_features = fea_extractor.extract_imgFea_from_street_block()

    #get street block img features

    img_features1 = np.load(r'./dataset/parcel_imgFea_thu.npy')
    #img_features1 = norm_feature(img_features1)
    # img_features = np.load(r'./dataset/parcel_imgFea_CFSplit32B_logits.npy')
    # img_features = norm_feature(img_features)
    # img_features = np.load(r'./dataset/parcel_imgFea_thu.npy')
    # img_features = norm_feature(img_features)

    # get street block poi features.
    poi_features = statistic_parcel_poi.main(r'./dataset/parcel_poi_infos_json.txt')
    poi_features = norm_feature(poi_features)

    #process mobike features
    #mobike_features = process_mobike()

    # get street block mobike features
    mobike_features = np.loadtxt(r'./dataset/parcel_mobike.txt')
    mobike_features = norm_feature(mobike_features)

    #cls.main([img_features1], norm=True)
    while 1:
        #cls.main([img_features], norm=False)
        cls.main([img_features1], norm = False)



