import json
import numpy as np

poi_type2label = {
    '地名地址信息': 0,
    '商务住宅': 1,
    '购物服务': 2,
    '生活服务': 3,
    '餐饮服务': 4,
    '科教文化服务': 5,
    '政府机构及社会团体': 6,
    '公司企业': 7,
    '体育休闲服务': 8,
    '汽车维修': 9,
    '住宿服务': 10,
    '医疗保健服务': 11,
    '金融保险服务': 12,
    '交通设施服务': 13,
    '汽车服务': 14,
    '风景名胜': 15,
    '摩托车服务': 16,
    '通行设施': 17,
    '汽车销售': 18,
}



def find_poi_types(path_parcel_poi_json_info):
    poi_types = []
    with open(path_parcel_poi_json_info, 'r') as f:
        infos = json.loads(f.read())['infos']
    for info in infos:
        parcel_index = info['parcel_index']
        print(parcel_index)
        poi_infos = info['poi_infos']
        for poi in poi_infos:
            base_type = poi['base_type']
            if base_type not in poi_types:
                poi_types.append(base_type)
    poi_index = 0
    for type in poi_types:
        line = "'%s': %d," % (type, poi_index)
        poi_index += 1
        print(line)


def statistic_poi_type_in_parcel(path_parcel_poi_json_info):
    # this function is to read parcel_poi_infos_json, return an array contains all poi features sequencely. every row contains a parcel' poi feature
    array_poi_statistic = np.zeros((1064, 19))
    with open(path_parcel_poi_json_info, 'r') as f:
        infos = json.loads(f.read())['infos']
    for info in infos:
        parcel_index = info['parcel_index']
        print("parcel_index:", parcel_index)
        poi_infos = info['poi_infos']
        print(len(poi_infos))
        for poi in poi_infos:
            base_type = poi['base_type']
            base_type_index = poi_type2label[base_type]
            array_poi_statistic[parcel_index][base_type_index] += 1
    # np.savetxt("statistic_poi_type.txt", array_poi_statistic)
    print(np.max(array_poi_statistic))
    return array_poi_statistic

def main(path = 'parcel_poi_infos_json.txt'):
    return statistic_poi_type_in_parcel(path)

if __name__ == '__main__':
    path_parcel_poi_info_json = r'parcel_poi_infos_json.txt'
    statistic_poi_type_in_parcel(path_parcel_poi_info_json)
