import MySQLdb
import geohash
import pandas as pd
import config
import csv
import numpy as np

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


def test_one_row_2db(mobike_csv_path):
    with open(mobike_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:]
        row = rows[0]
        print(row)
        order_id = row[0]
        user_id = row[1]
        bike_id = row[2]
        bike_type = row[3]
        start_time = row[4]
        hashed_start_loc = row[5]
        hashed_end_loc = row[6]

        date = start_time.split(' ')[0]
        hour = int(start_time.split(' ')[1].split(':')[0])
        is_weekend = date_is_weekend(date)

        start_ll = geohash.decode(hashed_start_loc)
        start_lng = float(start_ll[1])
        start_lat = float(start_ll[0])

        end_ll = geohash.decode(hashed_end_loc)
        end_lng = float(end_ll[1])
        end_lat = float(end_ll[0])

        sql_insert = "replace into `mobike` (`order_id`, `user_id`, `bike_id`, `bike_type`, `start_time`, `geohashed_start_loc`, `geohashed_end_loc`, `start_lat`, `start_lng`,`end_lat`,`end_lng`,`is_weekend`,`hour`) values"
        sql_insert += '("%s","%s","%s","%s","%s","%s","%s",%f,%f,%f,%f,%d,%d),' % (
            order_id, user_id, bike_id, bike_type, start_time, hashed_start_loc, hashed_end_loc, start_lat, start_lng,
            end_lat, end_lng, is_weekend, hour)
        print(sql_insert)
        db = MySQLdb.connect(**get_db_config())
        cursor = db.cursor()
        try:

            sql_insert = sql_insert[:-1]
            # with open('1.txt', 'w') as f:
            #     f.write(sql_insert)
            # print(str(sql_insert))
            cursor.execute(sql_insert)
            db.commit()
        except Exception as e:
            print(e)
            db.rollback()


def date_is_weekend(date):
    """
    判断摩拜单车数据中的日期是否是周末，数据中的日期范围是2017-05-10 到 2017-05-24，中间有两个周末
    :param date: 要判断的日期，格式为“2017-xx-xx”
    :return: 如过是周末返回1,否则返回0
    """
    if date == '2017-05-13' or date == '2017-05-14' or date == '2017-05-20' or date == '2017-05-21':
        return 1
    else:
        return 0


def import_mobike2db(mobike_csv_path):
    with open(mobike_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:]
        sql_insert = "insert into `mobike` (`order_id`, `user_id`, `bike_id`, `bike_type`, `start_time`, `geohashed_start_loc`, `geohashed_end_loc`, `start_lat`, `start_lng`,`end_lat`,`end_lng`,`is_weekend`,`hour`) values"

        count = 0
        for row in rows:
            count += 1
            if count < 3200000:
                continue
            print(count)

            if count >= 3300000:
                break
            order_id = row[0]
            user_id = row[1]
            bike_id = row[2]
            bike_type = row[3]
            start_time = row[4]
            hashed_start_loc = row[5]
            hashed_end_loc = row[6]

            date = start_time.split(' ')[0]
            hour = int(start_time.split(' ')[1].split(':')[0])
            is_weekend = date_is_weekend(date)

            start_ll = geohash.decode(hashed_start_loc)
            start_lng = float(start_ll[1])
            start_lat = float(start_ll[0])

            end_ll = geohash.decode(hashed_end_loc)
            end_lng = float(end_ll[1])
            end_lat = float(end_ll[0])
            sql_insert += '("%s","%s","%s","%s","%s","%s","%s",%f,%f,%f,%f,%d,%d),' % (
                order_id, user_id, bike_id, bike_type, start_time, hashed_start_loc, hashed_end_loc, start_lat,
                start_lng,end_lat, end_lng, is_weekend, hour)

        db = MySQLdb.connect(**get_db_config())
        cursor = db.cursor()
        try:

            sql_insert = sql_insert[:-1]
            with open('sql.txt', 'w') as f:
                f.write(sql_insert)
            # print(str(sql_insert))
            cursor.execute(sql_insert)
            db.commit()
            print("success")
        except Exception as e:
            print(e)
            db.rollback()



if __name__ == "__main__":
    # import_mobike2db(mobike_csv_file_path)
    stastic()
