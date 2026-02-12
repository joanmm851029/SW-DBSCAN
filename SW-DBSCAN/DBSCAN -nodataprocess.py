import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from dataprocess import coordinates
from evaluate import continuous_prob,precision,abnormal_prob,BSSdivISS
from sklearn.metrics import calinski_harabasz_score,silhouette_score
from sklearn.metrics import davies_bouldin_score


import pandas as pd
UNCLASSIFIED = 0
NOISE = -1
# N=4961
dists = None
alpha=2.5
belta=2.5

#  寻找以点point_id为中心，eps 为半径的圆内的所有点的id
def find_points_in_eps(point_id, eps, n,datas):
    global N
    # 计算数据点两两之间的距离
    def getDistanceMatrix(datas, i, j):
        dist=0
        vi = datas[i, :]
        vj = datas[j, :]
        dist= np.sqrt(alpha*((vi[0]-vj[0])**2)+belta*((vi[1]-vj[1])**2)+(vi[2]-vj[2])**2+(vi[3]-vj[3])**2+(vi[4]-vj[4])**2+(vi[5]-vj[5])**2
                      +(vi[6]-vj[6])**2+(vi[7]-vj[7])**2+(vi[8]-vj[8])**2+(vi[9]-vj[9])**2)
        return dist

    for i in range(1,n):
        if point_id + i<N and dists[point_id,point_id+i]==0:
            dists[point_id,point_id+i]=getDistanceMatrix(datas,point_id,point_id+i)
        if point_id +i>=N and dists[point_id, point_id+i-N]==0 :
            dists[point_id, point_id+i-N] = getDistanceMatrix(datas, point_id, point_id+i-N)
    if point_id + n<N:
        index = (dists[point_id, point_id :point_id + n ] <= eps)
    else:
        index1 = (dists[point_id, point_id:N] <= eps)
        index2=(dists[point_id, 0:point_id + n-N] <= eps)
    return (np.where(index == True)[0]+point_id).tolist() if point_id+n<N \
        else (np.where(index1 == True)[0]+point_id).tolist()+(np.where(index2 == True)[0]).tolist()

# 聚类扩展
def expand_cluster(labs, cluster_id, seeds, eps, min_points,datas,slide_n):
    i = 0
    while i < len(seeds):
        # 获取一个临近点
        Pn = seeds[i]
        # 如果该点被标记为NOISE 则重新标记
        if labs[Pn] == NOISE:
            labs[Pn] = cluster_id
        # 如果该点没有被标记过
        elif labs[Pn] == UNCLASSIFIED:
            # 进行标记，并计算它的临近点 new_seeds
            labs[Pn] = cluster_id
            new_seeds = find_points_in_eps(Pn, eps, slide_n,datas)

            # 如果 new_seeds 足够长则把它加入到seed 队列中
            if len(new_seeds) >= min_points:
                seeds = seeds + new_seeds

        i = i + 1


def dbscan(datas, eps, min_points):
    # 将所有点的标签初始化为UNCLASSIFIED
    n_points = datas.shape[0]
    labs = [UNCLASSIFIED] * n_points

    slide_n=5
    cluster_id = 0
    # 遍历所有点
    for point_id in range(0, n_points):
        # 如果当前点已经处理过了
        if not (labs[point_id] == UNCLASSIFIED):
            continue

        # 没有处理过则计算临近点
        seeds = find_points_in_eps(point_id, eps,slide_n,datas)

        # 如果临近点数量过少则标记为 NOISE
        if len(seeds) < min_points:
            labs[point_id] = NOISE
        else:
            # 否则就开启一轮簇的扩张
            cluster_id = cluster_id + 1
            # 标记当前点
            labs[point_id] = cluster_id
            expand_cluster(labs, cluster_id, seeds, eps, min_points,datas,slide_n)
    return labs, cluster_id


# 绘图
def draw_cluster(coordinates_x,coordinates_y, labs, n_cluster):
    plt.cla()

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, n_cluster)]

    for i, lab in enumerate(labs):
        if lab == NOISE:
            plt.scatter(coordinates_x[i],coordinates_y[i], s=16., color=(0, 0, 0))
        else:
            # if lab==0:
            #     plt.scatter(coordinates_x[i], coordinates_y[i], s=16., color=(0, 0, 1))
            # elif lab>=162:
            #     plt.scatter(coordinates_x[i], coordinates_y[i], s=16., color=(0, 0, 1))
            # else:
            plt.scatter(coordinates_x[i],coordinates_y[i], s=16., color=colors[lab-1])
    plt.show()


if __name__ == "__main__":
    # 火线坐标
    coordinates_x, coordinates_y = coordinates('coordinates.txt')
    # 数据
    df = pd.read_csv('data.csv')
    global N
    N = df.shape[0]
    dists = np.zeros([N, N])
    # 遍历每一行数据
    datas=df.to_numpy()
    datas[:,2:]/=10

    eps = 0.15
    min_points =4

    start=time.time()
    labs, cluster_id = dbscan(datas, eps=eps, min_points=min_points)
    end=time.time()
    print("labs of my dbscan")
    print(labs)
    # print(cluster_id)

    data_imp = []
    data_pos = []

    # 遍历每一行数据
    for index, row in df.iterrows():
        # 处理每一行数据
        data_imp.append(row[0])
        data_pos.append(row[1])

    indexes_imp = [index for index, val in enumerate(data_imp) if val == 0.2]
    indexes_pos1 =[index for index, val in enumerate(data_pos) if val == 0.1] # 火尾
    indexes_pos2 = [index for index, val in enumerate(data_pos) if val == 0.2] # 火翼
    indexes_pos3 = [index for index, val in enumerate(data_pos) if val == 0.3] # 火头
    indexes_pos1 =[item for item in indexes_pos1 if item not in indexes_imp]
    indexes_pos2= [item for item in indexes_pos2 if item not in indexes_imp]
    indexes_pos3 = [item for item in indexes_pos3 if item not in indexes_imp]

    #
    # datas[:, 1] /= 10
    start1 = time.time()
    db = DBSCAN(eps=0.1, min_samples=4).fit(datas)
    end1 = time.time()
    skl_labels = db.labels_

    print('改进DBSCAN运行时间：', end - start)
    print("改进DBSCAN连续率：", continuous_prob(labs))
    print("改进DBSCAN重要部位查准率",precision(indexes_imp,indexes_pos1,indexes_pos2,indexes_pos3,labs))
    print("改进DBSCAN火线火尾查准率",precision(indexes_pos1,indexes_pos2,indexes_pos3,indexes_imp,labs))
    print("改进DBSCAN火线火翼查准率",precision(indexes_pos2,indexes_pos1,indexes_pos3,indexes_imp,labs))
    print("改进DBSCAN火线火头查准率",precision(indexes_pos3,indexes_pos1,indexes_pos2,indexes_imp,labs))
    print("改进DBSCAN异常点率：",abnormal_prob(labs))


    # calinski_dbscan = calinski_harabasz_score(datas, labs)
    # davies_bouldin_dbscan=davies_bouldin_score(datas, labs)
    # silhouette_avg = silhouette_score(datas, labs)
    #
    # print("改进DBSCAN Silhouette Coefficient系数: ", silhouette_avg)
    # print("改进DBSCANcalinski_dbscan指数:", calinski_dbscan)
    # print("改进DBSCANdavies_bouldin_score指数:",davies_bouldin_dbscan)

    # mean_inner_class_distance=BSSdivISS(datas,labs,coordinates_x, coordinates_y,alpha,belta)
    mean_inter_class_distance, mean_inner_class_distance, bssdiviss = BSSdivISS(datas, labs, coordinates_x,
                                                                                coordinates_y, alpha, belta)
    print("改进DBSCAN的BSS",mean_inter_class_distance)
    print("改进DBSCAN的ISS", mean_inner_class_distance)
    print("改进DBSCAN的BSS/ISS", bssdiviss)
    draw_cluster(coordinates_x,coordinates_y, labs, cluster_id)

    #
    print("labs of sk-DBSCAN")
    list_skl_labels = skl_labels.tolist()
    print(list_skl_labels)
    print('DBSCAN运行时间：', end1 - start1)
    print("DBSCAN连续率：", continuous_prob(list_skl_labels))
    print("DBSCAN重要部位查准率", precision(indexes_imp, indexes_pos1, indexes_pos2, indexes_pos3, list_skl_labels))
    print("DBSCAN火线火尾查准率", precision(indexes_pos1, indexes_pos2, indexes_pos3, indexes_imp, list_skl_labels))
    print("DBSCAN火线火翼查准率", precision(indexes_pos2, indexes_pos1, indexes_pos3, indexes_imp, list_skl_labels))
    print("DBSCAN火线火头查准率", precision(indexes_pos3, indexes_pos1, indexes_pos2, indexes_imp, list_skl_labels))
    print("DBSCAN异常点率：", abnormal_prob(list_skl_labels))

    # calinski_dbscan1 = calinski_harabasz_score(datas, list_skl_labels)
    # davies_bouldin_dbscan1 = davies_bouldin_score(datas, list_skl_labels)
    # silhouette_avg1 = silhouette_score(datas, list_skl_labels)
    #
    # print("DBSCAN Silhouette Coefficient系数: ", silhouette_avg1)
    # print("DBSCANcalinski_dbscan指数:", calinski_dbscan1)
    # print("DBSCANdavies_bouldin_score指数:", davies_bouldin_dbscan1)

    mean_inter_class_distance1, mean_inner_class_distance1, bssdiviss1 = BSSdivISS(datas, list_skl_labels, coordinates_x,
                                                                                coordinates_y, alpha, belta)
    # mean_inner_class_distance1= BSSdivISS(datas, list_skl_labels,coordinates_x,coordinates_y, alpha, belta)
    print("DBSCAN的BSS", mean_inter_class_distance1)
    print("DBSCAN的ISS", mean_inner_class_distance1)
    print("DBSCAN的BSS/ISS", bssdiviss1)
    cluster_id= len(set(list_skl_labels)) - (1 if -1 in list_skl_labels else 0)
    draw_cluster(coordinates_x,coordinates_y, list_skl_labels, cluster_id)

    from sklearn.cluster import OPTICS

    start2 = time.time()
    clustering = OPTICS(min_samples=4, max_eps=0.15).fit(datas)
    end2 = time.time()
    labels_optics = clustering.labels_

    print("labs of  OPTICS")
    list_labels_optics = labels_optics.tolist()
    print(list_labels_optics)
    print('OPTICS运行时间：', end2 - start2)
    print("OPTICS连续率：", continuous_prob(list_labels_optics))
    print("OPTICS重要部位查准率", precision(indexes_imp, indexes_pos1, indexes_pos2, indexes_pos3, list_labels_optics))
    print("OPTICS火线火尾查准率", precision(indexes_pos1, indexes_pos2, indexes_pos3, indexes_imp, list_labels_optics))
    print("OPTICS火线火翼查准率", precision(indexes_pos2, indexes_pos1, indexes_pos3, indexes_imp, list_labels_optics))
    print("OPTICS火线火头查准率", precision(indexes_pos3, indexes_pos1, indexes_pos2, indexes_imp,list_labels_optics))
    print("OPTICS异常点率：", abnormal_prob(list_labels_optics))

    mean_inter_class_distance2, mean_inner_class_distance2, bssdiviss2 = BSSdivISS(datas, list_labels_optics, coordinates_x,
                                                                                coordinates_y, alpha, belta)
    # mean_inner_class_distance1= BSSdivISS(datas, list_skl_labels,coordinates_x,coordinates_y, alpha, belta)
    print("OPTICS的BSS", mean_inter_class_distance2)
    print("OPTICS的ISS", mean_inner_class_distance2)
    print("OPTICS的BSS/ISS", bssdiviss2)
    cluster_id = len(set(list_labels_optics)) - (1 if -1 in list_labels_optics else 0)
    draw_cluster(coordinates_x,coordinates_y, list_labels_optics, cluster_id)

    import hdbscan

    start3 = time.time()
    clusterer = hdbscan.HDBSCAN(min_samples=4)
    clusterer.fit(datas)
    end3 = time.time()
    labels_hdbscan = clusterer.labels_

    print("labs of  HDBSCAN")
    list_labels_hdbscan = labels_hdbscan.tolist()
    print(list_labels_hdbscan)
    print('HDBSCAN运行时间：', end3 - start3)
    print("HDBSCAN连续率：", continuous_prob(list_labels_hdbscan))
    print("HDBSCAN重要部位查准率", precision(indexes_imp, indexes_pos1, indexes_pos2, indexes_pos3, list_labels_hdbscan))
    print("HDBSCAN火线火尾查准率", precision(indexes_pos1, indexes_pos2, indexes_pos3, indexes_imp, list_labels_hdbscan))
    print("HDBSCAN火线火翼查准率", precision(indexes_pos2, indexes_pos1, indexes_pos3, indexes_imp, list_labels_hdbscan))
    print("HDBSCAN火线火头查准率", precision(indexes_pos3, indexes_pos1, indexes_pos2, indexes_imp, list_labels_hdbscan))
    print("HDBSCAN异常点率：", abnormal_prob(list_labels_hdbscan))

    mean_inter_class_distance3, mean_inner_class_distance3, bssdiviss3 = BSSdivISS(datas, list_labels_hdbscan,
                                                                                   coordinates_x,
                                                                                   coordinates_y, alpha, belta)
    # mean_inner_class_distance1= BSSdivISS(datas, list_skl_labels,coordinates_x,coordinates_y, alpha, belta)
    print("HDBSCAN的BSS", mean_inter_class_distance3)
    print("HDBSCAN的ISS", mean_inner_class_distance3)
    print("HDBSCAN的BSS/ISS", bssdiviss3)
    cluster_id = len(set(list_labels_hdbscan)) - (1 if -1 in list_labels_hdbscan else 0)
    draw_cluster(coordinates_x, coordinates_y, list_labels_hdbscan, cluster_id)




