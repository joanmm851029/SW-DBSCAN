import numpy as np

def continuous_prob(data):
    c_k=max(data)
    # 计算非连续率
    n1=0
    def is_list_continuous(data,lst):
        if len(lst)==1:
            return False
        else:
            segments = []
            start = lst[0]
            current = lst[0]
            for num in lst[1:]:
                if num == current + 1:
                    current = num
                else:
                    segments.append((start, current))
                    start = num
                    current = num
            segments.append((start, current))  # 添加最后一个段
            segments_revision=[]
            len_seg=len(segments)-1
            flag=[0]*len_seg
            for i in range(len_seg):
                len_s=segments[i+1][0]-segments[i][1]
                for j in range(1,len_s):
                    index_x=segments[i][1]+j
                    if data[index_x]!=-1:
                        flag[i]=1
                        break
            index_f= [index for index, value in enumerate(flag) if value == 1]
            if index_f:
                segments_revision.append((segments[0][0],segments[index_f[0]][1]))
                for i in range(1,len(index_f)):
                    end=index_f[i]
                    start=index_f[i-1]+1
                    segments_revision.append((segments[start][0],segments[end][1]))
                segments_revision.append((segments[index_f[-1]+1][0], segments[len_seg][1]))
            else:
                segments_revision.append((segments[0][0], segments[len_seg][1]))
            max_l=segments_revision[0][1]-segments_revision[0][0]
            s_l=segments_revision[0][1]-segments_revision[0][0]
            for i in range(1,len(segments_revision)):
                l=segments_revision[i][1]-segments_revision[i][0]
                s_l=s_l+l
                if l>max_l:
                    max_l=l
            if max_l>2 and max_l/s_l>0.8:
                return True
            else:
                return False

    xiabao=np.zeros((c_k+1,len(data)))
    for i in range(1,c_k+1):
        for j in range(len(data)):
             if data[j]==i:
                 xiabao[i][j]=1
    # max_i=0
    for i in range(1,c_k+1):
        index = (xiabao[i,:] == 1)
        index_num=(np.where(index == True)[0]).tolist()
        # if max_i<max(index_num):
        # print('c_'+str(i)+':')
        # print((np.where(index == True)[0]).tolist())
        #     max_i=max(index_num)
        if is_list_continuous(data,index_num)==True:
            n1=n1+1
    if c_k==0:
        return 1.0
    # print(n1)
    else:
        return n1/c_k

def isexist(list,data,n):
    n1=n
    while n1>0:
        if list[n1]!=list[n1-1]+1:
            break
        n1-=1
    for i in range(n1):
        if data[list[i]]==data[list[n]] :
            return False
    return True

# 计算查准率
#错误分类的个数
def wrong_num(list1,list2,list3,list4,data):
    flag1=0
    flag2=0
    Flag=0
    n=0
    for i in range(len(list1)):
        if data[list1[i]]!=-1:
            for j in range(len(list2)):
                if data[list1[i]]==data[list2[j]]:
                    Flag=1
                    flag1=1
                    break
            if flag1==0:
                for k in range(len(list3)):
                    a=data[list1[i]]
                    b=data[list3[k]]
                    if data[list1[i]]==data[list3[k]]:
                        Flag=1
                        flag2=1
                        break
            if flag1==0 and flag2==0:
                for m in range(len(list4)):
                    if data[list1[i]]==data[list4[m]]:
                        Flag=1
                        break
            c=i<len(list1)-1 and data[list1[i]]==data[list1[i+1]] and list1[i]+1!=list1[i+1]
            # if Flag==1 or isexist(list1,data,i) :
            if Flag == 1 or c:
                  n+=1
            flag1=0
            flag2=0
            Flag=0
    return n

def precision(list1, list2, list3, list4,data):  # list1:被统计的列表，list2，list3,list4：参照列表
    r_num=len(list1)-wrong_num(list1,list2,list3,list4,data)
    return r_num/len(list1)

def abnormal_prob(data):
    n1=len(data)
    n2=data.count(-1)
    return n2/n1


def find_indices(lst, element):
    return [i for i, x in enumerate(lst) if x == element]

def improve_euclidean_distance(x1, x2,alpha,belta):
    dist = np.sqrt(alpha * ((x1[0] - x2[0]) ** 2) + belta * ((x1[1] - x2[1]) ** 2) + (x1[2] - x2[2]) ** 2 + (
                x1[3] - x2[3]) ** 2 + (x1[4] - x2[4]) ** 2 + (x1[5] - x2[5]) ** 2
                   + (x1[6] - x2[6]) ** 2 + (x1[7] - x2[7]) ** 2 + (x1[8] - x2[8]) ** 2 + (x1[9] - x2[9]) ** 2+ (x1[10] - x2[10]) ** 2 + (x1[11] - x2[11]) ** 2)
    return dist

def euclidean_distance(x1, x2):
    dist = np.sqrt(((x1[0] - x2[0]) ** 2) +  ((x1[1] - x2[1]) ** 2) + (x1[2] - x2[2]) ** 2 + (
                x1[3] - x2[3]) ** 2 + (x1[4] - x2[4]) ** 2 + (x1[5] - x2[5]) ** 2
                   + (x1[6] - x2[6]) ** 2 + (x1[7] - x2[7]) ** 2 + (x1[8] - x2[8]) ** 2 + (x1[9] - x2[9]) ** 2)
    return dist
def Betweenclass(class1,class2,alpha,belta):
    inter_class_dists = []
    for i in range(len(class1)):
        for j in range(len(class2)):
            inter_class_dists.append(improve_euclidean_distance(class1[i], class2[j],alpha,belta))
            # inter_class_dists.append(euclidean_distance(class1[i], class2[j]))
    return inter_class_dists

def findnonzero(array1):
    return [i for i, row in enumerate(array1) if any(row)]

def BSSdivISS(datas,labs,coordinates_x, coordinates_y,alpha,belta):
    class_n=max(labs)
    feature_n=len(datas[0])
    classes=np.zeros((class_n+1,len(coordinates_x),feature_n+2))
    for i in range(1,class_n+1):
        index_i=find_indices(labs,i)
        for j in range(len(index_i)):
            for k in range(feature_n):
                classes[i][j][k]=datas[index_i[j]][k]
            classes[i][j][-2]=coordinates_x[index_i[j]]
            classes[i][j][-1] = coordinates_y[index_i[j]]
    inter_dists=[]
    class1=[]
    class2=[]
    inner_dists = []
    for i in range(1,class_n+1):
        nonzeroindex_i = np.array(findnonzero(classes[i]))
        inner_class_dists = Betweenclass(classes[i][nonzeroindex_i], classes[i][nonzeroindex_i], alpha, belta)
        inner_dists = inner_dists + inner_class_dists

        for j in range(i+1,class_n+1):
          nonzeroindex_j=np.array(findnonzero(classes[j]))
          inter_class_dists=Betweenclass(classes[i][nonzeroindex_i],classes[j][nonzeroindex_j],alpha,belta)
          inter_dists=inter_dists+inter_class_dists
    mean_inter_class_distance = np.mean(inter_dists)

    mean_inner_class_distance = np.mean(inner_dists)
    return mean_inter_class_distance,mean_inner_class_distance,mean_inter_class_distance/mean_inner_class_distance
    # return mean_inner_class_distance






