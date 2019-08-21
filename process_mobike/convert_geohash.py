import geohash
import pandas as pd

mobike_csv_file_path = r"/media/jsl/ubuntu/data/mobike/MOBIKE_CUP_2017/train.csv"
train_data = pd.read_csv(mobike_csv_file_path)
start_loc = train_data['starttime']
# with open('1.txt', 'w') as f:
#     for item in start_loc:
#         ll = geohash.decode(item)
#         lng = ll[1]
#         lat = ll[0]
#         f.write(lng+','+lat+'\r\n')

print(start_loc)