from sklearn.externals import joblib
from copy import deepcopy
import random
import pandas as pd
import sklearn
import os
import numpy as np


class KMeans():
    """
    Parameters
    ----------
    n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
    n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
    max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
    """

    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        max_iter=300
    ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(self, x):
        """
        用fit方法对数据进行聚类
        :param x: 输入数据
        :best_centers: 簇中心点坐标 数据类型: ndarray
        :best_labels: 聚类标签 数据类型: ndarray
        :return: self
        """
        ###################################################################################
        #### 请勿修改该函数的输入输出 ####
        ###################################################################################
        # #

        # #
        ###################################################################################
        ############# ?????? main ????, ???????? #############
        ###################################################################################

        best_centers = None
        best_labels = None

        best_shift = None

        for i in range(self.n_init):
            # init centers
            init_indexes = random.sample(range(len(x)), k=self.n_clusters)
            arr = x.to_numpy()
            init_centers = np.zeros([self.n_clusters, arr.shape[1]])
            for i in range(self.n_clusters):
                init_centers[i] = arr[init_indexes[i]]

            centers, labels, shift = self.kmeans(x, init_centers)
            if best_shift is None or shift < best_shift:
                best_centers = centers
                best_labels = labels
                best_shift = shift

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        return self

    def kmeans(self, x, centers_init, tol=2e-4):
        n_samples = x.shape[0]
        n_clusters = self.n_clusters
        x_numpy = x.to_numpy()
        centers = centers_init
        centers_new = np.zeros_like(centers)
        center_shift = np.zeros(n_clusters, dtype='float')
        labels = np.full(n_samples, -1, dtype=np.int32)
        labels_old = labels.copy()

        for _ in range(self.max_iter):
            center_shift = np.full(n_clusters, 0.)
            for i in range(n_samples):
                minDistance = np.inf
                for j in range(n_clusters):
                    tmpDist = self.distance(
                        x_numpy[i,].tolist(), centers[j,].tolist())
                    if tmpDist < minDistance:
                        minDistance = tmpDist
                        minIndex = j
                labels[i] = minIndex
            for i in range(n_clusters):
                cluster = np.extract(labels == i, x_numpy)
                centers_new[i] = np.mean(cluster, axis=0)
                center_shift[i] = self.distance(
                    centers_new[i].tolist(), centers[i].tolist())

            shift = (center_shift ** 2).sum() / n_clusters ** 0.5
            if shift <= tol or np.array_equal(labels, labels_old):
                break
            centers = np.copy(centers_new)
            labels_old = np.copy(labels)
        return centers, labels, shift

    def distance(self, v1, v2):
        d = 0
        for i in range(len(v1)):
            d += (v1[i] - v2[i]) ** 2
        d **= 0.5
        return d

kmeans = KMeans(n_clusters=5, n_init=50, max_iter=800)
kmeans.fit(data)

# 保存模型，用于后续的测试打分
joblib.dump(kmeans, './results/model.pkl')
joblib.dump(scaler, './results/scaler.pkl')
joblib.dump(pca, './results/pca.pkl')


def preprocess_data(df):
    """
    数据处理及特征工程等
    :param df: 读取原始 csv 数据，有 timestamp、cpc、cpm 共 3 列特征
    :return: 处理后的数据, 返回 pca 降维后的特征
    """
    # 请使用joblib函数加载自己训练的 scaler、pca 模型，方便在测试时系统对数据进行相同的变换
    # ====================数据预处理、构造特征等========================
    # 例如
    # df['hours'] = df['timestamp'].dt.hour
    # df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    # ========================  模型加载  ===========================
    # 请确认需要用到的列名，e.g.:columns = ['cpc','cpm']

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['cpc X cpm'] = df['cpc'] * df['cpm']
    df['cpc / cpm'] = df['cpc'] / df['cpm']

    columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm']
    data = df[columns]
    # 例如
    scaler = joblib.load('./results/scaler.pkl')
    pca = joblib.load('./results/pca.pkl')

    data = scaler.fit_transform(data)
    data = pca.fit_transform(data)
    data = pd.DataFrame(
        data, columns=['Dimension' + str(i+1) for i in range(pca.n_components)])

    return data


def get_distance(data, kmeans, n_features):
    """
    计算样本点与聚类中心的距离
    :param data: preprocess_data 函数返回值，即 pca 降维后的数据
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return:每个点距离自己簇中心的距离，Series 类型
    """
    # ====================计算样本点与聚类中心的距离========================
    distance = []
    for i in range(len(data)):
        point = np.array(data.iloc[i, :n_features])
        center = kmeans.cluster_centers_[kmeans.labels_[i], :n_features]
        distance.append(np.linalg.norm(point - center))

    return pd.Series(distance)


def get_anomaly(data, kmean, ratio):
    """
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点

    :param data: preprocess_data 函数返回值，即 pca 降维后的数据，DataFrame 类型
    :param kmean: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间，float 类型
    :return: data 添加 is_anomaly 列，该列数据是根据阈值距离大小判断每个点是否是异常值，元素值为 False 和 True
    """
    # ====================检验出样本中的异常点========================
    num_anomaly = int(len(data) * ratio)
    data['distance'] = get_distance(data, kmean, n_features=len(data.columns))
    data_copy = deepcopy(data)
    sorted_data = data_copy['distance'].sort_values(ascending=False)
    boundary = sorted_data.reset_index(drop=True)[num_anomaly]

    is_anomaly = []
    for i in range(len(data)):
        if data_copy.at[i, 'distance'] < boundary:
            is_anomaly.append(False)
        else:
            is_anomaly.append(True)
    data['is_anomaly'] = is_anomaly
    return data


def predict(preprocess_data):
    """
    该函数将被用于测试，请不要修改函数的输入输出，并按照自己的模型返回相关的数据。
    在函数内部加载 kmeans 模型并使用 get_anomaly 得到每个样本点异常值的判断
    :param preprocess_data: preprocess_data函数的返回值，一般是 DataFrame 类型
    :return:is_anomaly:get_anomaly函数的返回值，各个属性应该为（Dimesion1,Dimension2,......数量取决于具体的pca），distance,is_anomaly，请确保这些列存在
            preprocess_data:  即直接返回输入的数据
            kmeans: 通过joblib加载的对象
            ratio:  异常点的比例，ratio <= 0.03   返回非异常点得分将受到惩罚！
    """
    data = deepcopy(preprocess_data)
    # 异常值所占比率
    ratio = 0.03
    # 加载模型
    kmeans = joblib.load('./results/model.pkl')
    # 获取异常点数据信息
    is_anomaly = get_anomaly(data, kmeans, ratio)

    return is_anomaly, preprocess_data, kmeans, ratio
