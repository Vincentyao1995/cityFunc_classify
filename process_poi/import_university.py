import config
import MySQLdb
import ogr
import os
import numpy as np


def get_db_config():
    db_config = {}
    db_config['host'] = config.db_host
    db_config['port'] = config.db_port
    db_config['user'] = config.db_user
    db_config['passwd'] = config.db_pw
    db_config['db'] = config.db_name
    db_config['charset'] = 'utf8'
    return db_config


def import_poi_to_db(poi_path, t_name):
    """
    import poi file to the db
    :param poi_path: poi shapefile's path
    :param t_name: the table's name which save the poi
    :return: 
    """
    sql_insert = "replace into `%s` (`lat`, `lng`) values" % t_name
    ds_poi = ogr.Open(poi_path)
    i = 0
    for layer in ds_poi:
        for feature in layer:
            i += 1
            geom = feature.GetGeometryRef()

            lat = geom.GetY()
            lng = geom.GetX()
            print(lat, lng)

            sql_insert += '(%f,%f),' % (lat, lng)

    db = MySQLdb.connect(**get_db_config())
    cursor = db.cursor()
    try:

        sql_insert = sql_insert[:-1]
        with open('1.txt', 'w') as f:
            f.write(sql_insert)
        # print(str(sql_insert))
        cursor.execute(sql_insert)
        db.commit()
    except Exception as e:
        print(e)
        db.rollback()


def statistic_parcel_school():
    parcel_clip_gcj02_path = r'/media/jsl/ubuntu/data/parcel/haidian_clip_parcels_gcj02.shp'
    ds_parcel = ogr.Open(parcel_clip_gcj02_path)
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
    parcel_university = np.zeros((n_parcels, 1))
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
        sql_select = "select `lat`,`lng` from `university` where `lat` > %f and `lat` < %f and `lng` > %f and `lng` < %f" % (
            lat_min, lat_max, lng_min, lng_max)
        n = cursor.execute(sql_select)
        print("retangular:", n)
        if n == 0:
            count += 1
            continue
        universities = cursor.fetchall()
        dates = []
        record_count = 0
        for item in universities:
            lat = item[0]
            lng = item[1]
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(lng, lat)
            if parcel_geom.Contains(point):
                record_count += 1
                parcel_university[index][0] += 1

        print("inner:", record_count)
    np.savetxt("parcel_euniversity.txt", parcel_university)


if __name__ == "__main__":
    university_path = r"/media/jsl/ubuntu/data/school/高校——84.shp"
    # import_poi_to_db(university_path, "university")
    statistic_parcel_school()