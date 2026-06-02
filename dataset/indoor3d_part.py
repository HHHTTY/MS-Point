import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset


def load_data_partseg(partition):
    # download_shapenetpart()
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg




def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        # self.partseg_colors = load_color_partseg()
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]
    
    


class S3DISDatasetHDF5(Dataset):
    """Chopped Scene"""

    def __init__(self, root, split='train', test_area=5):
        if root is None:
            BASE_DIR = ''
            DATA_DIR = os.path.join(BASE_DIR, 'data')
            root = 'data/'
        self.root = root
        self.all_files = self.getDataFiles(os.path.join(self.root, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))
        self.room_filelist = self.getDataFiles(os.path.join(self.root, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))
        self.scene_points_list = []
        self.semantic_labels_list = []
        for h5_filename in self.all_files:
            data_batch, label_batch = self.loadh5DataFile(os.path.join(self.root, h5_filename))
            self.scene_points_list.append(data_batch)
            self.semantic_labels_list.append(label_batch)
        self.data_batches = np.concatenate(self.scene_points_list, 0)
        self.label_batches = np.concatenate(self.semantic_labels_list, 0)
        test_area = 'Area_' + str(test_area)
        train_idxs, test_idxs = [], []
        for i, room_name in enumerate(self.room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)
        assert split in ['train', 'test']
        if split == 'train':
            self.data_batches = self.data_batches[train_idxs, ...]
            self.label_batches = self.label_batches[train_idxs]
        else:
            self.data_batches = self.data_batches[test_idxs, ...]
            self.label_batches = self.label_batches[test_idxs]

    @staticmethod
    def getDataFiles(list_filename):
        return [line.rstrip() for line in open(list_filename)]

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __getitem__(self, index):
        points = self.data_batches[index, :]
        labels = self.label_batches[index].astype(np.int32)
        return points, labels

    def __len__(self):
        return len(self.data_batches)