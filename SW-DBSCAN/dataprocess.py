import math
import rioxarray
import pyproj
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
import random
def coordinates(file):
    with open(file, 'r',encoding='utf-8') as file:
        content = file.read()
    content_list=[i for i in content.split()]
    coordinates=[]
    for i in range(len(content_list)):
        coordinates.append(tuple(list(map(float, content_list[i].split(",")))))

    def tupletoxylist(x):
        x_list = []
        y_list = []
        for i in range(len(x)):
            x_list.append(x[i][0])
            y_list.append(x[i][1])
        return x_list, y_list

    coordinates_x,coordinates_y=tupletoxylist(coordinates)
    plt.scatter(coordinates_x, coordinates_y, color='b')

    return coordinates_x,coordinates_y

def wind_coordinate(file):
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    wind_coordinates_list = []
    for line in lines:
        if '<coordinates>' in line:
            row_list = line.split('coordinates')
            wind_coordinates_list.append(row_list[1].strip('>').strip('</').split(','))
    wind_coordinates_x_list = []
    wind_coordinates_y_list = []
    for i in range(len(wind_coordinates_list)):
        wind_coordinates_x_list.append(wind_coordinates_list[i][0])
        wind_coordinates_y_list.append(wind_coordinates_list[i][1])
    return list(map(float, wind_coordinates_x_list)),list(map(float, wind_coordinates_y_list))

def nearest_cordinates_wind(coordinates_x,coordinates_y,wind_coordinates_x,wind_coordinates_y):
    distant=[]
    nearest_cordinates=[]
    for i in range(len(coordinates_x)):
        for j in range(len(wind_coordinates_x)):
            distant.append(math.sqrt((coordinates_x[i] - wind_coordinates_x[j]) ** 2 + (coordinates_y[i] - wind_coordinates_y[j]) ** 2))
        nearest_cordinates.append(distant.index(min(distant)))
        distant=[]
    # plt.scatter([wind_coordinates_x[index] for index in nearest_cordinates], [wind_coordinates_y[index] for index in nearest_cordinates], color='b')

    return [wind_coordinates_x[index] for index in nearest_cordinates],[wind_coordinates_y[index] for index in nearest_cordinates]

def ascprocess(file):
    pop = rioxarray.open_rasterio('firelineintensity.tif')
    utm_crs = pyproj.CRS('epsg:26910')
    wgs84_crs = pyproj.CRS('epsg:4326')
    transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs)
    latitude_x = []
    longitude_y = []
    for j in range(len(pop.y)):
        for i in range(len(pop.x)):
            # 调用函数进行转换
            latitude, longitude = transformer.transform(pop.x[i], pop.y[j])
            latitude_x.append(latitude)
            longitude_y.append(longitude)
    return latitude_x,longitude_y

def nearest_cordinates_asc(coordinates_x,coordinates_y,latitude_asc,longitude_asc,coordinate_list):
    distant = []
    nearest_cordinates = []
    index_tmp=[]
    for i in range(len(coordinates_x)):
        for j in range(len(coordinate_list)):
            index_lati_long=coordinate_list[j][0]*791+coordinate_list[j][1]
            distant.append(math.sqrt(
                (coordinates_x[i] - longitude_asc[index_lati_long]) ** 2 + (coordinates_y[i] - latitude_asc[index_lati_long]) ** 2))
            index_tmp.append(index_lati_long)
        near_i=index_tmp[distant.index(min(distant))]//791
        near_j=index_tmp[distant.index(min(distant))]%791
        nearest_cordinates.append((near_i,near_j))
        distant = []
        index_tmp=[]
    index_list=[t[0]*791+t[1] for t in nearest_cordinates]
    # plt.scatter([longitude_asc[index] for index in index_list], [latitude_asc[index] for index in index_list], color='b')

    return nearest_cordinates

def readasc(file):
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    i = 0
    asc_data = []
    for line in lines:
        if i > 5:
            asc_data.append(list(map(float, line.strip('\n').split(' ')[1:])))
        i = i + 1
    coordinate_list = []
    for i in range(len(asc_data)):
        for j in range(len(asc_data[i])):
            if asc_data[i][j] != -9999.0:
                coordinate_list.append((i, j))
    return coordinate_list

def check_output_coordinate(l_list,nearest_cordinates_wind_x,nearest_cordinates_wind_y,w):
    index=[]
    wind_speed=0
    wind_direction=0
    spread_direction=0
    coord_list=l_list[7].lstrip('      <Point><coordinates>').rstrip('</coordinates></Point>\n').split(',')
    for index_x, value in enumerate(nearest_cordinates_wind_x):
            if value==float(coord_list[0]):
                if float(coord_list[1])==nearest_cordinates_wind_y[index_x]:
                    if w==1:
                        wind_speed=float(l_list[5].lstrip('\t\t<SimpleData name="mph">').rstrip('</SimpleData>\n'))
                        wind_direction=float(l_list[4].lstrip('\t\t<SimpleData name="AMAP_AZI">').rstrip('</SimpleData>\n'))
                    else:
                        spread_direction=float(l_list[4].lstrip('\t\t<SimpleData name="AMAP_AZI">').rstrip('</SimpleData>\n'))
                    index.append(index_x)
    if w==1:
        return wind_speed,wind_direction,index
    else:
        return spread_direction,index


def wind_speed_direction_premier(file,nearest_cordinates_wind_x,nearest_cordinates_wind_y,w):
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    i=0
    index_tmp=0
    wind_speed_dict= {}
    wind_direction_dict={}
    spread_direction_dict={}
    for line in lines:
        if '<Placemark>' in line:
            index_tmp = i
            i = i + 1
            continue
        if '</Placemark>' in line:
            if w == 1:
                w_speed, w_direction,index = check_output_coordinate(lines[index_tmp:i],nearest_cordinates_wind_x,nearest_cordinates_wind_y,w)
                for index_dict, value in enumerate(index):
                    wind_speed_dict[value] = w_speed
                    wind_direction_dict[value] = w_direction
            else:
                s_direction,index=check_output_coordinate(lines[index_tmp:i],nearest_cordinates_wind_x,nearest_cordinates_wind_y,w)
                for index_dict, value in enumerate(index):
                    spread_direction_dict[value] = s_direction
            i=i+1
            continue
        i=i+1
    if w==1:
        return wind_speed_dict,wind_direction_dict
    else:
        return spread_direction_dict

def asc_premier(file,nearest_cordinates):
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    asc_matrix=[]
    x_dict={}
    for i in range(6,len(lines)):
        asc_matrix.append(list(map(float, lines[i].rstrip(' ').lstrip(' ').split(' '))))
    for index,value in enumerate(nearest_cordinates):
       x_dict[index] =asc_matrix[value[0]][value[1]]
    return x_dict

def get_column_and_target_indices(matrix, col_index, target_value):
    """
    提取二维列表的指定列，并找出该列中值为目标值的元素的所有行索引
    :param matrix: 输入的二维列表
    :param col_index: 目标列的索引（从0开始）
    :param target_value: 要匹配的目标值（可任意类型：int/str/float等）
    :return: (target_column: 提取的目标列列表, target_indices: 列中值为目标值的行索引列表)
    """
    # 步骤1：提取目标列（处理行长度不足的情况，填充None避免索引越界）
    target_column = []
    for row in matrix:
        if col_index < len(row):
            target_column.append(row[col_index])
        else:
            target_column.append(None)  # 行长度不足时填充None，可根据需求改为其他值（如-1）

    # 步骤2：遍历目标列，记录值等于目标值的索引
    target_indices = []
    for idx, value in enumerate(target_column):
        if value == target_value:
            target_indices.append(idx)

    return target_indices

def  dataprocess():
    #火线坐标
    coordinates_x,coordinates_y=coordinates('coordinates.txt')
    #风矢量坐标
    wind_coordinates_x,wind_coordinates_y=wind_coordinate('wind.txt')
    #挑选火线上的风矢量坐标
    nearest_cordinates_wind_x,nearest_cordinates_wind_y=nearest_cordinates_wind(coordinates_x, coordinates_y, wind_coordinates_x, wind_coordinates_y)
    #火线上的风速和风向
    wind_speed_dict,wind_direction_dict=wind_speed_direction_premier('wind.txt',nearest_cordinates_wind_x,nearest_cordinates_wind_y,1)
    #蔓延方向
    spread_direction_dict = wind_speed_direction_premier('spreadvector.txt', nearest_cordinates_wind_x,
                                                                        nearest_cordinates_wind_y,0)
    #asc文件坐标
    latitude_asc, longitude_asc=ascprocess('firelineintensity.tif')
    coordinate_list=readasc('firelineintensity.asc')
    # 挑选火线上的asc坐标(i,j)
    nearest_cordinates=nearest_cordinates_asc(coordinates_x,coordinates_y,latitude_asc,longitude_asc,coordinate_list)
    plt.show()
    #火强度
    fireintensity_dict=asc_premier('firelineintensity.asc',nearest_cordinates)
    #aspect坡向
    aspect_dict=asc_premier('aspect.asc',nearest_cordinates)
    # 森林郁闭度
    canopycover_dict=asc_premier('canopycover.asc',nearest_cordinates)
    #可燃物类型
    fuelmode_dict=asc_premier('fuel model.asc',nearest_cordinates)
    # slope坡度
    slope_dict = asc_premier('slope.asc', nearest_cordinates)
    # 上下山火
    updown_fire_dict={}
    sorted_spread_direction_dict={k: v for k, v in sorted(spread_direction_dict.items(), key=lambda item: item[0])}
    for key, value in sorted_spread_direction_dict.items():
        a=aspect_dict[key]
        if abs(value-aspect_dict[key])<90 or abs(value-aspect_dict[key])>270:
            updown_fire_dict[key]=0.2   #下坡
        else:
            updown_fire_dict[key] = 0.1   #上坡
    # 火头、火翼、火尾
    head_tail_wind_dict={}
    sorted_wind_direction_dict = {k: v for k, v in sorted(wind_direction_dict.items(), key=lambda item: item[0])}
    for key, value in sorted_spread_direction_dict.items():
        if abs(value -sorted_wind_direction_dict[key]) < 30 or abs(value - sorted_wind_direction_dict[key]) > 330:
            head_tail_wind_dict[key] = 0.3    #火头
        elif abs(value - sorted_wind_direction_dict[key]) > 150 and abs(value - sorted_wind_direction_dict[key]) < 210:
            head_tail_wind_dict[key] = 0.1  #火尾
        else:
            head_tail_wind_dict[key] = 0.2   #火翼
    datas=np.zeros([len(coordinates_x),10])
    important_aim =np.zeros([len(coordinates_x)])+0.1
    #产生重要目标
    # random_integers = [random.randint(1, len(coordinates_x)-20) for _ in range(10)]
    random_integers = [115, 832, 1293, 612,384, 1430]
    for i in range(len(random_integers)):
        important_aim[random_integers[i]:random_integers[i]+20]=0.2

    sorted_wind_speed_dict = {k: v for k, v in sorted(wind_speed_dict.items(), key=lambda item: item[0])}
    wind_direction_array= np.array(list(sorted_wind_direction_dict.values()))
    wind_speed_array=np.array(list(sorted_wind_speed_dict.values()))
    fireintensity_array = np.array(list(fireintensity_dict.values()))
    aspect_array = np.array(list(aspect_dict.values()))
    slope_array = np.array(list(slope_dict.values()))
    canopycover_array = np.array(list(canopycover_dict.values()))
    fuelmode_array = np.array(list(fuelmode_dict.values()))
    updown_fire_array = np.array(list(updown_fire_dict.values()))
    head_tail_wind_array = np.array(list(head_tail_wind_dict.values()))

    def min_max_normalize(data):
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data=[]
        eps=1e-10
        for i in range(len(data)):
            # if max_val - min_val!=0:
            #     normalized_data.append((data[i] - min_val) / (max_val - min_val))
            # else:
            normalized_data.append((data[i] - min_val+eps) / (max_val - min_val+eps))
        return normalized_data


    datas[:, 0] = important_aim  #重点目标
    datas[:,1]=head_tail_wind_array     #火头（尾翼）
    datas[:, 2] = min_max_normalize(wind_direction_array)       #风向
    datas[:,3]=min_max_normalize(fireintensity_array)  #火强度
    datas[:, 4] = min_max_normalize(aspect_array)   #坡向
    datas[:,5]=min_max_normalize(slope_array)   #坡度
    datas[:, 6] = min_max_normalize(canopycover_array)  # 森林郁闭度
    datas[:, 7] = min_max_normalize(fuelmode_array)  #可燃物类型
    datas[:, 8] = min_max_normalize(updown_fire_array)  #上下火
    datas[:, 9] = min_max_normalize(wind_speed_array)  #风速
    # datas[:, 0] = important_aim  # 重点目标
    # datas[:, 1] = head_tail_wind_array  # 火头（尾翼）
    # datas[:, 2] = wind_direction_array  # 风向
    # datas[:, 3] = fireintensity_array  # 火强度
    # datas[:, 4] = aspect_array  # 坡向
    # datas[:, 5] = slope_array  # 坡度
    # datas[:, 6] = canopycover_array  # 森林郁闭度
    # datas[:, 7] = fuelmode_array  # 可燃物类型
    # datas[:, 8] = updown_fire_array  # 上下火
    # datas[:, 9] = wind_speed_array  # 风速
    # scaler = MinMaxScaler()
    new_row= np.array([[0, 0, 0,0,0,0,0,0,0,0]])
    datas_new=np.vstack((new_row, datas))
    #  拟合并转换数据
    #datas_scaled = scaler.fit_transform(datas)
    np.savetxt('data.csv', datas_new, delimiter=',', fmt='%.6f')
    return datas,coordinates_x,coordinates_y

if __name__ == '__main__':
    datas,coordinates_x,coordinates_y=dataprocess()
    plt.show
