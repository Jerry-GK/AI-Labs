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
    n_clusters ָ������Ҫ����ĸ����������������Ҫ�Լ���������Ӱ������Ч��
    n_init ָ������������㷨����������һ���ͷ��ؽ�����������ж�κ󷵻���õ�һ�ν����n_init��ָ�����еĴ���
    max_iter ָ���������������ĵ���������������ǰ����������ֹͣ����
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
        ��fit���������ݽ��о���
        :param x: ��������
        :best_centers: �����ĵ����� ��������: ndarray
        :best_labels: �����ǩ ��������: ndarray
        :return: self
        """
        ###################################################################################
        #### �����޸ĸú������������ ####
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

# ����ģ�ͣ����ں����Ĳ��Դ��
joblib.dump(kmeans, './results/model.pkl')
joblib.dump(scaler, './results/scaler.pkl')
joblib.dump(pca, './results/pca.pkl')


def preprocess_data(df):
    """
    ���ݴ����������̵�
    :param df: ��ȡԭʼ csv ���ݣ��� timestamp��cpc��cpm �� 3 ������
    :return: ����������, ���� pca ��ά�������
    """
    # ��ʹ��joblib���������Լ�ѵ���� scaler��pca ģ�ͣ������ڲ���ʱϵͳ�����ݽ�����ͬ�ı任
    # ====================����Ԥ��������������========================
    # ����
    # df['hours'] = df['timestamp'].dt.hour
    # df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    # ========================  ģ�ͼ���  ===========================
    # ��ȷ����Ҫ�õ���������e.g.:columns = ['cpc','cpm']

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['cpc X cpm'] = df['cpc'] * df['cpm']
    df['cpc / cpm'] = df['cpc'] / df['cpm']

    columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm']
    data = df[columns]
    # ����
    scaler = joblib.load('./results/scaler.pkl')
    pca = joblib.load('./results/pca.pkl')

    data = scaler.fit_transform(data)
    data = pca.fit_transform(data)
    data = pd.DataFrame(
        data, columns=['Dimension' + str(i+1) for i in range(pca.n_components)])

    return data


def get_distance(data, kmeans, n_features):
    """
    ������������������ĵľ���
    :param data: preprocess_data ��������ֵ���� pca ��ά�������
    :param kmeans: ͨ�� joblib ���ص�ģ�Ͷ��󣬻���ѵ���õ� kmeans ģ��
    :param n_features: ���������Ҫ������������
    :return:ÿ��������Լ������ĵľ��룬Series ����
    """
    # ====================������������������ĵľ���========================
    distance = []
    for i in range(len(data)):
        point = np.array(data.iloc[i, :n_features])
        center = kmeans.cluster_centers_[kmeans.labels_[i], :n_features]
        distance.append(np.linalg.norm(point - center))

    return pd.Series(distance)


def get_anomaly(data, kmean, ratio):
    """
    ����������е��쳣�㣬�����Ϊ True �� False��True ��ʾ���쳣��

    :param data: preprocess_data ��������ֵ���� pca ��ά������ݣ�DataFrame ����
    :param kmean: ͨ�� joblib ���ص�ģ�Ͷ��󣬻���ѵ���õ� kmeans ģ��
    :param ratio: �쳣����ռȫ�����ݵİٷֱ�,�� 0 - 1 ֮�䣬float ����
    :return: data ��� is_anomaly �У����������Ǹ�����ֵ�����С�ж�ÿ�����Ƿ����쳣ֵ��Ԫ��ֵΪ False �� True
    """
    # ====================����������е��쳣��========================
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
    �ú����������ڲ��ԣ��벻Ҫ�޸ĺ���������������������Լ���ģ�ͷ�����ص����ݡ�
    �ں����ڲ����� kmeans ģ�Ͳ�ʹ�� get_anomaly �õ�ÿ���������쳣ֵ���ж�
    :param preprocess_data: preprocess_data�����ķ���ֵ��һ���� DataFrame ����
    :return:is_anomaly:get_anomaly�����ķ���ֵ����������Ӧ��Ϊ��Dimesion1,Dimension2,......����ȡ���ھ����pca����distance,is_anomaly����ȷ����Щ�д���
            preprocess_data:  ��ֱ�ӷ������������
            kmeans: ͨ��joblib���صĶ���
            ratio:  �쳣��ı�����ratio <= 0.03   ���ط��쳣��÷ֽ��ܵ��ͷ���
    """
    data = deepcopy(preprocess_data)
    # �쳣ֵ��ռ����
    ratio = 0.03
    # ����ģ��
    kmeans = joblib.load('./results/model.pkl')
    # ��ȡ�쳣��������Ϣ
    is_anomaly = get_anomaly(data, kmeans, ratio)

    return is_anomaly, preprocess_data, kmeans, ratio
