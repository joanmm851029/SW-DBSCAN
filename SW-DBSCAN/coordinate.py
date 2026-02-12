import matplotlib.pyplot as plt
import numpy as np
import math
from dataprocess import wind_coordinate,nearest_cordinates_wind
with open('coordinates-fire5.txt', 'r',encoding='utf-8') as file:
    content = file.read()
content_list=[i for i in content.split()]
coordinates=[]
for i in range(len(content_list)):
    coordinates.append(tuple(list(map(float, content_list[i].split(",")))))
# print(coordinates)

def tupletoxylist(x):
    x_list = []
    y_list = []
    for i in range(len(x)):
        x_list.append(x[i][0])
        y_list.append(x[i][1])
    return x_list, y_list

coordinates_x,coordinates_y=tupletoxylist(coordinates)

print(coordinates_x)
print(coordinates_y)

#经纬度转化为距离
def lat_lon_dist(x1,y1,x2,y2):
    lat1 = math.radians(y1)
    lat2 = math.radians(y2)
    lon1 = math.radians(x1)
    lon2 = math.radians(x2)
    a = math.cos(lat1) * math.cos(lat2) * math.cos(abs(lon2 - lon1))
    b = math.sin(lat1) * math.sin(lat2)
    dist=6371 * math.sqrt(abs(2 - 2 * (a + b)))
    return dist


dist=[]
dist_a=0
for i in range(len(coordinates_x)-1):
    di=lat_lon_dist(coordinates_x[i],coordinates_y[i],coordinates_x[i + 1],coordinates_y[i+1])
    dist.append(di)
    dist_a=dist_a+di
print(dist_a)


min_x_value, min_x_index = min((value, index) for index, value in enumerate(coordinates_x))
max_x_value, max_x_index = max((value, index) for index, value in enumerate(coordinates_x))
min_y_value, min_y_index = min((value, index) for index, value in enumerate(coordinates_y))
max_y_value, max_y_index = max((value, index) for index, value in enumerate(coordinates_y))
dist_x=lat_lon_dist(min_x_value,coordinates_y[min_x_index],max_x_value,coordinates_y[max_x_index])
dist_y=lat_lon_dist(min_y_value,coordinates_y[min_y_index],max_y_value,coordinates_y[max_y_index])
print((min_x_value,coordinates_y[min_x_index]),(max_x_value,coordinates_y[max_x_index]),(min_y_value,coordinates_y[min_y_index]),
      (max_y_value,coordinates_y[max_y_index]))
print(dist_x,dist_y)
x_list=[]
y_list=[]
for i in range(len(coordinates_x)):
    x_list.append(dist_x*(coordinates_x[i]-min_x_value)/(max_x_value-min_x_value))
    y_list.append(dist_y * (coordinates_y[i] - min_y_value) / (max_y_value - min_y_value))

# 创建一个画布和两个子图
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # figsize参数可以调整画布大小

# 在第一个子图上绘制第一个散点图
axs[0].scatter(np.array(coordinates_x), np.array(coordinates_y), color='red')
axs[0].set_xlabel('X:longitude')
axs[0].set_ylabel('Y:latitude')
axs[0].legend()

# 在第二个子图上绘制第二个散点图
axs[1].scatter(np.array(x_list), np.array(y_list), color='blue')
axs[1].set_xlabel('X:km')
axs[1].set_ylabel('Y:km')
axs[1].legend()

# 显示图形
plt.tight_layout()  # 自动调整子图参数，使之填充整个图形区域
plt.show()

plt.scatter(np.array(x_list), np.array(y_list), color='blue')
plt.xlabel('X(km)')
plt.ylabel('Y(km)')
plt.show()

wind_coordinates_x, wind_coordinates_y = wind_coordinate('wind.txt')
plt.scatter(wind_coordinates_x, wind_coordinates_y, color='g')
plt.scatter(coordinates_x, coordinates_y, color='r')
nearest_cordinates_wind_x, nearest_cordinates_wind_y = nearest_cordinates_wind(coordinates_x, coordinates_y,
                                                                               wind_coordinates_x, wind_coordinates_y)
plt.show()

