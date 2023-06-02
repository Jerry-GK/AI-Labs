import warnings
# ���Ӿ���
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition
# 1.�������ݲ��������ݴ���
# ���ݼ�·��
data_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/"

# 2.�����Ԥѵ��ģ�ͣ������Ԥѵ��ģ�ͣ����û������Ҫ����
def letterbox_image(image, size):
    """
    ����ͼƬ�ߴ�
    :param image: ����ѵ����ͼƬ
    :param size: ��Ҫ���������������ͼƬ�ߴ�
    :return: ���ؾ���������ͼƬ
    """
    new_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return new_image

def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.1):
    """
    ���ݴ�����
    :param data_path: ����·��
    :param height:�߶�
    :param width: ���
    :param batch_size: ÿ�ζ�ȡͼƬ������
    :param test_split: ���Լ����ֱ���
    :return: 
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # �������ˮƽ��ת
        T.RandomVerticalFlip(0.1),  # ���������ֱ��ת
        T.ToTensor(),  # ת��Ϊ����
        T.Normalize([0], [1]),  # ��һ��
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    # �������ݼ�
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # ����һ�� DataLoader ����
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return train_data_loader, valid_data_loader

pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
onet_path = "./torch_py/MTCNN/weights/onet.npy"
torch.set_num_threads(8)

# 3.����ģ�ͺ�ѵ��ģ�ͣ�ѵ��ģ��ʱ������ģ�ͱ����� results �ļ���
# ���� MobileNet ��Ԥѵ��ģ��Ȩ
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=32)
modify_x, modify_y = torch.ones((32, 3, 160, 160)), torch.ones((32))
epochs = 3
model = MobileNetV1(classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # �Ż���
print('�������...')

# ѧϰ���½��ķ�ʽ��acc���β��½����½�ѧϰ�ʼ���ѵ����˥��ѧϰ��
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=3)
# ��ʧ����
criterion = nn.CrossEntropyLoss()
# ѵ��ģ��
best_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
loss_list = []  # �洢��ʧ����ֵ
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in tqdm(enumerate(train_data_loader, 1)):
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)
        # print(pred_y.shape)
        # print(y.shape)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss
        loss_list.append(loss)
    print('step:' + str(epoch + 1) + '/' + str(epochs) + ' || Total Loss: %.4f' % (loss) + '\n')
# 4.����ģ�ͣ����Լ���Ϊ���ģ�ͱ����� result �ļ��У�����ģ�ͱ�������Ŀ�������ļ��У��������ӿ����ͨ����
torch.save(model.state_dict(), './results/temp.pth')
print('Finish Training.')