# -*- utf-8 -*-
import ogr
import osr
import config
import MySQLdb

'''
this file is to process poi data, process poi data and assign every parcel/streetblock POIs. result file is parcel_poi_infos_json.txt
'''

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
    sql_insert = "replace into `%s` (`id`, `name`, `city`, `district`, `type`, `type_code`, `base_type`, `sub_type`, `lat`, `lng`) values" % t_name
    ds_poi = ogr.Open(poi_path)
    i = 0
    for layer in ds_poi:
        for feature in layer:
            i += 1
            # if the poi's cunt is too many ,it will occur error, so reduce the number though the index i
            # if i <= 300000:
            #     continue
            # print(i)
            # if i > 300000:
            #     break
            geom = feature.GetGeometryRef()
            id = feature.GetField('id')
            name = feature.GetField('name')
            city = feature.GetField('city')
            district = feature.GetField('district')
            type = feature.GetField('type')
            type_code = feature.GetField('typecode')
            base_type = feature.GetField('basetype')
            sub_type = feature.GetField('subtype')
            lat = geom.GetY()
            lng = geom.GetX()
            if district == '海淀区':
                sql_insert += '("%s","%s","%s","%s","%s",%d,"%s","%s",%f,%f),' % (
                    id, name, city, district, type, type_code, base_type, sub_type, lat, lng)

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

def export_poi_to_txt(t_poi_name, txt_path):
    db = MySQLdb.connect(**get_db_config())
    cursor = db.cursor()
    sql_select_lat_lng = "select `lat`,`lng`,`name` from %s;" % t_name_poi
    cursor.execute(sql_select_lat_lng)
    pois = cursor.fetchall()
    poi_str = ''
    for poi in pois:
        poi_str += "{0},{1},{2}\r\n".format(poi[0], poi[1], poi[2])
    with open(txt_path, 'w') as f:
        f.write(poi_str)


def cal_parcel_contain_poi(parcel_path, poi_t_name):
    ds_parcel = ogr.Open(parcel_path)
    num_layers = ds_parcel.GetLayerCount()
    layer = ds_parcel.GetLayerByIndex(0)
    spatial_ref = layer.GetSpatialRef()
    # print(spatial_ref)
    test_str = ''

    db = MySQLdb.connect(**get_db_config())
    cursor = db.cursor()
    count = 0
    print(layer.GetFeatureCount())

    parcel_poi_infos = []
    index = 0

    for parcel in layer:
        parcel_poi_info = {}
        index = parcel.GetField('Id')
        parcel_poi_info['parcel_index'] = index
        print(index)
        poi_infos = []
        parcel_geom = parcel.GetGeometryRef()
        env = parcel_geom.GetEnvelope()
        # print(env)
        lat_min = env[2]
        lat_max = env[3]
        lng_min = env[0]
        lng_max = env[1]
        sql_select = "select `lat`, `lng`, `name`, `base_type`, `id` from `%s` where `lat` > %f and `lat` < %f and `lng` > %f and `lng` < %f" % (poi_t_name, lat_min, lat_max, lng_min, lng_max)
        n = cursor.execute(sql_select)
        if n == 0:
            count += 1
            continue
        pois = cursor.fetchall()
        for poi in pois:
            lat = poi[0]
            lng = poi[1]
            name = poi[2]
            base_type = poi[3]
            poi_id = poi[4]
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(lng, lat)
            poi_info = {}
            if parcel_geom.Contains(point):
                poi_info['name'] = name
                poi_info['base_type'] = base_type
                poi_info['id'] = poi_id
                poi_infos.append(poi_info)
        parcel_poi_info['poi_infos'] = poi_infos
        parcel_poi_infos.append(parcel_poi_info)

    with open('parcel_poi_infos.txt', 'w') as f:
        f.write(str(parcel_poi_infos))

def test_contain(parcel_path):
    ds_parcel = ogr.Open(parcel_path)
    layer = ds_parcel.GetLayerByIndex(0)
    feature = layer[0]
    geom = feature.GetGeometryRef()
    env = geom.GetEnvelope()
    print((env[0]+env[1])/2)
    print((env[2]+env[3])/2)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(116.1979867875891, 39.921506110056384)
    print(point)
    print(geom.Contains(point))

if __name__ == '__main__':
    parcel_clip_gcj02_path = r'/media/jsl/ubuntu/data/parcel/haidian_clip_parcels_gcj02.shp'
    # parcel_path = r'/media/jsl/ubuntu/data/city_data/haidian_data/haidian_parcels.shp'
    poi_path_hd_gcj02_shp = r'/media/jsl/ubuntu/data/POI/poi_haidian_raw.shp'
    t_name_poi = 'poi_haidian'
    # import_poi_to_db(poi_path_hd_gcj02_shp, 'poi_haidian')
    # export_poi_to_txt(t_name_poi, 'poi_txt.txt')
    # cal_parcel_contain_poi(poi_hd_wgs_arcgis, t_name_poi)
    cal_parcel_contain_poi(parcel_clip_gcj02_path, t_name_poi)
    # test_contain(parcel_clip_path)
