from copy import deepcopy
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
import os
import sklearn
import warnings
import copy
import random
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
# matplotlib inline

file_dir = './data'
csv_files = os.listdir(file_dir)
print(csv_files)

# df ��Ϊ�������� DataFrame ��ʼ��Ϊ��
df = pd.DataFrame()
feature = ['cpc', 'cpm']
df_features = []
for col in feature:
    infix = col + '.csv'
    path = os.path.join(file_dir, infix)
    df_feature = pd.read_csv(path)
    # �����������洢�������ں�������
    df_features.append(df_feature)

# 2 �� DataFrame ��ʱ������
df = pd.merge(left=df_features[0], right=df_features[1])
df.head()

# ��ȡ df ������Ϣ
df.info()

df.describe()

# �� timestamp ��ת��Ϊʱ������
df['timestamp'] = pd.to_datetime(df['timestamp'])

# �� df ���ݰ�ʱ���������򣬷�������չʾ
df = df.sort_values(by='timestamp').reset_index(drop=True)
df.head()

# ����ʱ������� cpc �� cpm ָ������
fig, axes = plt.subplots(2, 1)
df.plot(kind='line', x='timestamp', y='cpc', figsize=(
    20, 10), ax=axes[0], title='cpc', color='g')
df.plot(kind='line', x='timestamp', y='cpm', figsize=(
    20, 10), ax=axes[1], title='cpm', color='b')
plt.tight_layout()

sns.heatmap(df.corr(), cmap='coolwarm', annot=True)

# ���� cpc �� cpm ��ɢ��ͼ�����к������� cpc���������� cpm
plt.scatter(x=df['cpc'], y=df['cpm'], alpha=0.5)
# �������ݼ����ĵ㣬�������� cpc ��ƽ��ֵ���������� cpm ��ƽ��ֵ
plt.scatter(x=df['cpc'].mean(), y=df['cpm'].mean(), c='red', alpha=0.8)


def simple_distance(data):
    """
    ���㵱ǰ�㣨cpc��cpm������cpc_mean��cpm_mean���ļ��ξ��루L2 ������
    :param data: ataDFrame ����cpc��cpm ��
    :return: Series��ÿһ��cpc��cpm��ƽ��ֵֵ��ľ����С
    """
    mean = np.array([data['cpc'].mean(), data['cpm'].mean()])
    distance = []
    for i in range(0, len(data)):
        point = np.array(data.iloc[i, 1:3])
        # ��ǰ�㣨cpc��cpm����ƽ��ֵ�㣨cpc_mean��cpm_mean��֮��ļ��ξ��루L2 ������
        distance.append(np.linalg.norm(point - mean))
    distance = pd.Series(distance)
    return distance


df['distance'] = simple_distance(df[df.columns])
df.head()

df.describe()

# ����ʱ������� cpc �� cpm ָ������
fig, axes = plt.subplots(3, 1)
df.plot(kind='line', x='timestamp', y='cpc',
        figsize=(20, 10), ax=axes[0], title='cpc')
df.plot(kind='line', x='timestamp', y='cpm',
        figsize=(20, 10), ax=axes[1], title='cpm')
df.plot(kind='line', x='timestamp', y='distance',
        figsize=(20, 10), ax=axes[2], title='distance')
plt.tight_layout()

ratio = 0.005
num_anomaly = int(len(df) * ratio)
threshould = df['distance'].sort_values(
    ascending=False).reset_index(drop=True)[num_anomaly]
print('��ֵ���룺' + str(threshould))

df['is_anomaly'] = df['distance'].apply(lambda x: x > threshould)
df.head()

normal = df[df['is_anomaly'] == 0]
anormal = df[df['is_anomaly'] == 1]
plt.scatter(x=normal['cpc'], y=normal['cpm'], c='blue', alpha=0.5)
plt.scatter(x=anormal['cpc'], y=anormal['cpm'], c='red', alpha=0.5)

scaler = StandardScaler()
df[['cpc', 'cpm']] = scaler.fit_transform(df[['cpc', 'cpm']])
df.describe()

# �����쳣���ݱ���
ratio = 0.005
num_anomaly = int(len(df) * ratio)
df['distance2'] = simple_distance(df)
threshould = df['distance2'].sort_values(
    ascending=False).reset_index(drop=True)[num_anomaly]
print('��ֵ���룺'+str(threshould))
df['is_anomaly2'] = df['distance2'].apply(lambda x: x > threshould)
normal = df[df['is_anomaly2'] == 0]
anormal = df[df['is_anomaly2'] == 1]
plt.scatter(x=normal['cpc'], y=normal['cpm'], c='blue', alpha=0.5)
plt.scatter(x=anormal['cpc'], y=anormal['cpm'], c='red', alpha=0.5)

a = df.loc[df['is_anomaly2'] == 1, ['timestamp', 'cpc']]
plt.figure(figsize=(20, 10))
plt.plot(df['timestamp'], df['cpc'], color='blue')
plt.scatter(a['timestamp'], a['cpc'], color='red')
plt.show()

# ������������Թ�ϵ
df['cpc X cpm'] = df['cpm'] * df['cpc']
df['cpc / cpm'] = df['cpc'] / df['cpm']

# ���Ի�ȡʱ���ϵ
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

df1 = df[['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm', 'hours', 'daylight']]
df1.corr()

sns.heatmap(df1.corr(), cmap='coolwarm', annot=True)


# �ڽ��������任֮ǰ�ȶԸ����������б�׼��
columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm']
data = df[columns]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=columns)

# ͨ�� n_components ָ����Ҫ���͵���ά��
n_components = 3
pca = PCA(n_components=n_components)
data = pca.fit_transform(data)
data = pd.DataFrame(data, columns=['Dimension' + str(i+1)
                    for i in range(n_components)])
data.head()

# �����任�ռ䣨�������󣩣���������ָ����n_components = k��ֵ��ѡ�񷽲����� k ��ֵ����Ӧ�ĵ�����������ɵ���������
print(pd.DataFrame(pca.components_, columns=[
      'Dimension' + str(i + 1) for i in range(pca.n_features_)]))

# ÿ����������ռ���������ķ���ٷֱ�
pca.explained_variance_ratio_

# ���� n �������ķ���
pca.explained_variance_

var_explain = pca.explained_variance_ratio_
# �����ۼƺͣ�axis=0���������ۼӡ�axis=1���������ۼӡ�axis����������ֵ���Ͱ����鵱��һ��һά����
cum_var_explian = np.cumsum(var_explain)

plt.figure(figsize=(10, 5))
plt.bar(range(len(var_explain)), var_explain, alpha=0.3,
        align='center', label='the interpreted independently variance ')
plt.step(range(len(cum_var_explian)), cum_var_explian, where='mid',
         label='the cumulative interpretation variance')
plt.ylabel('the explain variance rate')
plt.xlabel('PCA')
plt.legend(loc='best')
plt.show()


fig = plt.figure(1, figsize=(6, 6))
ax = Axes3D(fig)
ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], edgecolors='k')


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
        max_iter=400,
        tol=2e-4
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

        best_centers = None
        best_labels = None
        best_shift = None

        for i in range(self.n_init):
            # init centers
            init_indexes = random.sample(range(len(x)), k=self.n_clusters)
            print(i)
            arr = x.to_numpy()
            init_centers = np.zeros([self.n_clusters, arr.shape[1]])
            for i in range(self.n_clusters):
                init_centers[i] = arr[init_indexes[i]]

            centers, labels, shift = self.kmeans(x, init_centers, self.tol)
            if best_shift is None or shift < best_shift:
                best_centers = centers
                best_labels = labels
                best_shift = shift

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        return self

    def kmeans(self, x, centers_init, tol):
        n_samples = x.shape[0]
        n_clusters = self.n_clusters
        x_numpy = x.to_numpy()
        centers = centers_init
        centers_new = np.zeros_like(centers)
        center_shift = np.zeros(n_clusters, dtype='float')
        labels = np.full(n_samples, -1, dtype=np.int32)
        labels_old = labels.copy()

        for p in range(self.max_iter):
            # print(p)
            center_shift = np.full(n_clusters, 0.)
            for i in range(n_samples):
                minDistance = np.inf
                for j in range(n_clusters):
                    temp = self.edistance(
                        x_numpy[i,].tolist(), centers[j,].tolist())
                    if temp < minDistance:
                        minDistance = temp
                        minIndex = j
                labels[i] = minIndex
            for i in range(n_clusters):
                cluster = np.extract(labels == i, x_numpy)
                centers_new[i] = np.mean(cluster, axis=0)
                center_shift[i] = self.edistance(
                    centers_new[i].tolist(), centers[i].tolist())

            shift = (center_shift ** 2).sum() / n_clusters ** 0.5
            if np.array_equal(labels, labels_old) or shift <= tol:
                break
            centers = np.copy(centers_new)
            labels_old = np.copy(labels)
        return centers, labels, shift

    def edistance(self, v1, v2):
        d = 0
        for i in range(len(v1)):
            d += (v1[i] - v2[i]) ** 2
        d **= 0.5
        return d


print("keans finished")
kmeans = KMeans(n_clusters=5, n_init=50, max_iter=1000)  # may take a long time
kmeans.fit(data)

labels = pd.Series(kmeans.labels_)
fig = plt.figure(1, figsize=(6, 6))
ax = Axes3D(fig)
ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:,
           2], c=labels.astype('float'), edgecolors='k')


score1 = calinski_harabasz_score(data, kmeans.labels_)
score2 = silhouette_score(data, kmeans.labels_)

print('calinski_harabasz_score:', score1)
print('silhouette_score:', score2)

# Ѱ����Ѿ�����Ŀ

'''
n_clusters ָ������Ҫ����ĸ����������������Ҫ�Լ���������Ӱ������Ч��
init ָ����ʼ�������ĵ�ĳ�ʼ����ʽ��kmeans++��һ�ֳ�ʼ����ʽ��������ѡ��Ϊrandom
n_init ָ������������㷨����������һ���ͷ��ؽ�����������ж�κ󷵻���õ�һ�ν����n_init��ָ�����еĴ���
max_iter ָ���������������ĵ���������������ǰ����������ֹͣ����
'''
score1_list = []
score2_list = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, n_init=10, max_iter=200)
    kmeans.fit(data)
    score1 = round(calinski_harabasz_score(data, kmeans.labels_), 2)
    score2 = round(silhouette_score(data, kmeans.labels_), 2)
    score1_list.append(score1)
    score2_list.append(score2)
    print('������Ŀ:%s  calinski_harabasz_score:%-10s  silhouette_score:%-10s' %
          (i, score1, score2))


def get_distance(data, kmeans, n_features):
    """
    ������뺯��
    :param data: ѵ�� kmeans ģ�͵�����
    :param kmeans: ѵ���õ� kmeans ģ��
    :param n_features: ���������Ҫ������������
    :return: ÿ��������Լ������ĵľ���
    """
    distance = []
    for i in range(0, len(data)):
        point = np.array(data.iloc[i, :n_features])
        center = kmeans.cluster_centers_[kmeans.labels_[i], :n_features]
        distance.append(np.linalg.norm(point - center))
    distance = pd.Series(distance)
    return distance


ratio = 0.01
num_anomaly = int(len(data) * ratio)
new_data = deepcopy(data)
new_data['distance'] = get_distance(
    new_data, kmeans, n_features=len(new_data.columns))
threshould = new_data['distance'].sort_values(
    ascending=False).reset_index(drop=True)[num_anomaly]
print('��ֵ���룺'+str(threshould))

# ������ֵ�����С�ж�ÿ�����Ƿ����쳣ֵ
new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)
normal = new_data[new_data['is_anomaly'] == 0]
anormal = new_data[new_data['is_anomaly'] == 1]
new_data.head()

fig = plt.figure(1, figsize=(6, 6))
ax = Axes3D(fig)
ax.scatter(anormal.iloc[:, 0], anormal.iloc[:, 1],
           anormal.iloc[:, 2], c='red', edgecolors='k')
ax.scatter(normal.iloc[:, 0], normal.iloc[:, 1],
           normal.iloc[:, 2], c='blue', edgecolors='k')

# ����ģ�ͣ����ں����Ĳ��Դ��
joblib.dump(kmeans, './results/model.pkl')
joblib.dump(scaler, './results/scaler.pkl')
joblib.dump(pca, './results/pca.pkl')

a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpc']]
plt.figure(figsize=(20, 6))
plt.plot(df['timestamp'], df['cpc'], color='blue')
# ����� cpc ���쳣��
plt.scatter(a['timestamp'], a['cpc'], color='red')
plt.show()

a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpm']]
plt.figure(figsize=(20, 6))
plt.plot(df['timestamp'], df['cpm'], color='blue')
# ����� cpm ���쳣��
plt.scatter(a['timestamp'], a['cpm'], color='red')
plt.show()


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

    df['cpc X cpm'] = df['cpc'] * df['cpm']
    df['cpc / cpm'] = df['cpc'] / df['cpm']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

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
