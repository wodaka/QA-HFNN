# import packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.cluster import KMeans
import torch

#
class_names = [name[26:] for name in glob.glob('./data/scene15/data/train/*')]
# print(class_names)
class_names = dict(zip(range(0,len(class_names)), class_names))
# print(class_names)

def load_dataset(path, num_per_class=-1):
    data = []
    labels = []
    for id, class_name in class_names.items():
        img_path_class = glob.glob(path + class_name + '/*.jpg')
        if num_per_class > 0:
            img_path_class = img_path_class[:num_per_class]
        labels.extend([id]*len(img_path_class))
        for filename in img_path_class:
            data.append(cv2.imread(filename, 0))
    return data, labels

# compute dense SIFT
def computeSIFT(data):
    x = []
    for i in range(0, len(data)):
        sift = cv2.xfeatures2d.SIFT_create()
        img = data[i]
        step_size = 15
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        dense_feat = sift.compute(img, kp)
        x.append(dense_feat[1])
    return x


# build BoW presentation from SIFT of training images
def clusterFeatures(all_train_desc, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_train_desc)
    return kmeans

# form training set histograms for each training image using BoW representation
def formTrainingSetHistogram(x_train, kmeans, k):
    train_hist = []
    for i in range(len(x_train)):
        data = copy.deepcopy(x_train[i])
        predict = kmeans.predict(data)
        train_hist.append(np.bincount(predict, minlength=k).reshape(1, -1).ravel())

    return np.array(train_hist)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_dataset_scene15_first():
    # load training dataset
    train_data, train_label = load_dataset('data/scene15/data/train/', 100)
    train_num = len(train_label)

    # load testing dataset
    test_data, test_label = load_dataset('data/scene15/data/test/', 100)
    test_num = len(test_label)

    # print(train_num,test_num)
    # extract dense sift features from training images
    x_train = computeSIFT(train_data)
    x_test = computeSIFT(test_data)

    all_train_desc = []
    for i in range(len(x_train)):
        for j in range(x_train[i].shape[0]):
            all_train_desc.append(x_train[i][j,:])

    all_train_desc = np.array(all_train_desc)

    k = 200
    kmeans = clusterFeatures(all_train_desc, k)

    # form training and testing histograms
    train_hist = formTrainingSetHistogram(x_train, kmeans, k)
    test_hist = formTrainingSetHistogram(x_test, kmeans, k)

    # normalize histograms
    scaler = preprocessing.StandardScaler().fit(train_hist)
    train_hist = scaler.transform(train_hist)
    test_hist = scaler.transform(test_hist)

    train_dataset = CustomDataset(train_hist, train_label)
    test_dataset = CustomDataset(test_hist, test_label)
    torch.save(train_dataset, './data/scene15/train_dataset.pth')
    torch.save(test_dataset, './data/scene15/test_dataset.pth')

    return train_dataset,test_dataset

def load_dataset_scene15():
    train_dataset = torch.load('./data/scene15/train_dataset.pth')
    test_dataset = torch.load('./data/scene15/test_dataset.pth')
    return train_dataset,test_dataset