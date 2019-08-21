import MySQLdb
import geohash
import pandas as pd
import config
import csv
import numpy as np
import matplotlib.pyplot as plt
import ogr

mobike_csv_file_path = r"/media/jsl/ubuntu/data/mobike/MOBIKE_CUP_2017/train.csv"


def get_db_config():
    db_config = {}
    db_config['host'] = config.db_host
    db_config['port'] = config.db_port
    db_config['user'] = config.db_user
    db_config['passwd'] = config.db_pw
    db_config['db'] = config.db_name
    db_config['charset'] = 'utf8'
    return db_config


def stastic_zgc():
    sql = "select hour,start_time from mobike WHERE end_lat > 39.976035 and end_lat < 39.986621 and end_lng > 116.301189 AND end_lng < 116.317792 and is_weekend=1"
    db = MySQLdb.connect(**get_db_config())
    cursor = db.cursor()
    cursor.execute(sql)
    hours = cursor.fetchall()
    bike_times = np.zeros((24))
    dates = []
    for item in hours:
        h = item[0]
        date = str(item[1]).split(" ")[0]
        if date not in dates:
            dates.append(date)
        bike_times[h] += 1

    bike_times = bike_times / len(dates)
    np.savetxt("zgc_end_weekend.txt", bike_times)


def stastic_resident():
    sql = "select hour,start_time from mobike WHERE start_lat > 40.062275 and start_lat < 40.079615 and start_lng > 116.421304 AND start_lng < 116.431046 and is_weekend=0"
    db = MySQLdb.connect(**get_db_config())
    cursor = db.cursor()
    cursor.execute(sql)
    hours = cursor.fetchall()
    bike_times = np.zeros((24))
    dates = []
    for item in hours:
        h = item[0]
        date = str(item[1]).split(" ")[0]
        if date not in dates:
            dates.append(date)
        bike_times[h] += 1

    bike_times = bike_times / len(dates)
    np.savetxt("res_start_week.txt", bike_times)


def draw(statistic_file1, statistic_file2):
    bike_times_week = np.loadtxt(statistic_file1)
    bike_times_weekend = np.loadtxt(statistic_file2)
    print(bike_times_week)
    bike_times_week = (bike_times_week - np.min(bike_times_week)) / (np.max(bike_times_week) - np.min(bike_times_week))
    print(bike_times_week)
    bike_times_weekend = (bike_times_weekend - np.min(bike_times_weekend)) / (
        np.max(bike_times_weekend) - np.min(bike_times_weekend))
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21', '22', '23']
    x = range(len(names))
    y1 = bike_times_week
    y2 = bike_times_weekend
    plt.plot(x, y1, marker='o', mec='r', mfc='w', label=u'res')
    plt.plot(x, y2, marker='*', ms=10, label=u'com')
    plt.legend()  # 让图例生效
    plt.show()


def statisitc_haidian_parcel(parcel_path):
    ds_parcel = ogr.Open(parcel_path)
    num_layers = ds_parcel.GetLayerCount()
    layer = ds_parcel.GetLayerByIndex(0)
    spatial_ref = layer.GetSpatialRef()
    # print(spatial_ref)
    test_str = ''

    db = MySQLdb.connect(**get_db_config())
    cursor = db.cursor()
    count = 0
    # the number of parcels
    n_parcels = layer.GetFeatureCount()

    index = 0
    parcel_mobike = np.zeros((n_parcels, 24))
    for parcel in layer:
        index = parcel.GetField('Id')
        print("index:", index)
        parcel_geom = parcel.GetGeometryRef()
        env = parcel_geom.GetEnvelope()
        # print(env)
        lat_min = env[2]
        lat_max = env[3]
        lng_min = env[0]
        lng_max = env[1]
        sql_select = "select `end_lat`,`end_lng`,`hour`,`start_time` from `mobike` where `end_lat` > %f and `end_lat` < %f and `end_lng` > %f and `end_lng` < %f and `is_weekend` = 0" % (
            lat_min, lat_max, lng_min, lng_max)
        n = cursor.execute(sql_select)
        print("retangular:", n)
        if n == 0:
            count += 1
            continue
        bikes = cursor.fetchall()
        dates = []
        record_count = 0
        for item in bikes:
            lat = item[0]
            lng = item[1]
            h = item[2]
            date = str(item[3]).split(" ")[0]
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(lng, lat)
            if parcel_geom.Contains(point):
                record_count+=1
                parcel_mobike[index][h] += 1
                if date not in dates:
                    dates.append(date)
        parcel_mobike[index] = parcel_mobike[index]/len(dates)
        print("inner:", record_count)
    np.savetxt("parcel_end_week.txt", parcel_mobike)


def convert_nan2zero():
    """
    去除NAN
    :return: 
    """
    a = np.loadtxt("parcel_end_week.txt")
    a = np.nan_to_num(a)
    np.savetxt("parcel_end_week_new.txt", a)

def nan_statistic():
    """
    mobike array normlization
    :return: 
    """
    a = np.loadtxt("parcel_start_week.txt")
    n = a.shape[0]
    for i in range(n):
        a[i] = (a[i]-min(a[i]))/(max(a[i])-min(a[i]))
    a = np.nan_to_num(a)
    np.savetxt("parcel_start_week_norm.txt", a)

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

    #concate four 8-period mobike features to one.
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

if __name__ == "__main__":
    # stastic_zgc()
    # stastic_resident()

    # ------绘制对比图------
    # statistic_file_path1 = "res_start_week.txt"
    # statistic_file_path2 = "zgc_start_week.txt"
    # draw(statistic_file_path1, statistic_file_path2)

    # ------统计每一个parcle中的mobike------
    # parcel_clip_wgs84_path = r'/media/jsl/ubuntu/data/parcel/haidian_clip_parcels_wgs84.shp'
    # statisitc_haidian_parcel(parcel_clip_wgs84_path)
    #nan_statistic()
    process_mobike()