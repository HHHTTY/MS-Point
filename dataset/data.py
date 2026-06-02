import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image
from .plyfile import load_ply
from . import data_utils as d_utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,TensorDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scipy.spatial.distance import cdist

IMG_DIR = 'data/ShapeNetRendering'


trans_1 = transforms.Compose(
            [
                
                d_utils.PointcloudRandomCrop(),
                # d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

    # 0114
trans_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                # d_utils.PointcloudRandomCutout(ratio_min=0.3, ratio_max=0.6),
                d_utils.PointcloudRandomCutout(ratio_min=0.2, ratio_max=0.7),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])
# 0127
trans_4 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                # d_utils.PointcloudRandomCutout(ratio_min=0.3, ratio_max=0.6),
                # d_utils.PointcloudRandomCutout(ratio_min=0.05, ratio_max=0.9),
                d_utils.PointcloudRandomCutout(ratio_min=0.05, ratio_max=0.8),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

trans_5 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                # d_utils.PointcloudRandomCutout(ratio_min=0.3, ratio_max=0.6),
                # d_utils.PointcloudRandomCutout(ratio_min=0.05, ratio_max=0.9),
                d_utils.PointcloudRandomCutout(ratio_min=0.4, ratio_max=0.55),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

trans_3 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudRandomCutout(ratio_min=0.15, ratio_max=0.8),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])


def load_modelnet_data(partition):
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_ScanObjectNN(partition):
    BASE_DIR = 'data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label

def load_shapenet_data():
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_filepath = []

    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs
        
    return all_filepath

def get_render_imgs(pcd_path):
    path_lst = pcd_path.split('/')
    # path_lst: ['data', 'ShapeNet', '02958343', '10f667965cd701eeeeae8bcbf655eede.ply']

    path_lst[1] = 'ShapeNetRendering'

    # path_lst[-1][:-4]: 10f667965cd701eeeeae8bcbf655

    path_lst[-1] = path_lst[-1][:-4]

    # path_lst = ['data', 'ShapeNetRendering', '02958343', '10f667965cd701eeeeae8bcbf655eede']
    path_lst.append('rendering')
    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))

    return img_path_list



class ShapeNetRender(Dataset):
    def __init__(self, img_transform=None,img_transform1=None, n_imgs=1):
        self.data = load_shapenet_data()
        self.transform = img_transform
        self.transform1 = img_transform1
        self.n_imgs = n_imgs
    
    def farthest_point_sampling(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 1024:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def farthest_point_sampling1(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 512:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def __getitem__(self, item):
        pcd_path = self.data[item]

        # render_img_path = random.choice(get_render_imgs(pcd_path))
        render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)

        render_img_list = []
        for indexing in range(0,1):
            render_img_path = render_img_path_list[indexing]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform(render_img_a)
            render_img_list.append(render_img_a)

        for indexing1 in range(1,3):
            render_img_path = render_img_path_list[indexing1]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform1(render_img_a)
            render_img_list.append(render_img_a)

        # render_img = Image.open(render_img_path).convert('RGB')
        # render_img = self.transform(render_img)  #.permute(1, 2, 0)
        # render_img_list.append(render_img)

        pointcloud_1 = load_ply(self.data[item])
        pointcloud_orig = pointcloud_1.copy()

        point_t1 = trans_2(pointcloud_orig)
        
#         1024
        # pointcloud_orig1 = d_utils.fps(torch.from_numpy(pointcloud_orig).float(),1024)
        pointcloud_orig1 = self.farthest_point_sampling(pointcloud_orig)
        point_t2 = trans_3(pointcloud_orig1)
        
        # point_t3 = d_utils.fps(torch.from_numpy(pointcloud_orig).float(),512)
        point_t3 = self.farthest_point_sampling1(pointcloud_orig1)
        
        # pointcloud = (pointcloud_orig, point_t1, point_t2)
        # pointcloud = (point_t1, point_t2)
        # pointcloud=pointcloud_1

        # imgs = (render_img_list[0], render_img_list[1], render_img_list[2],render_img_list[3])
        imgs = (render_img_list[0], render_img_list[1], render_img_list[2])

        return (point_t1, point_t2,point_t3), imgs  # render_img # render_img_list

    def __len__(self):
        return len(self.data)

class ShapeNetRender5(Dataset):
    def __init__(self, img_transform=None,img_transform1=None, n_imgs=1):
        self.data = load_shapenet_data()
        self.transform = img_transform
        self.transform1 = img_transform1
        self.n_imgs = n_imgs
    
    def farthest_point_sampling(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 1024:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def farthest_point_sampling1(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 512:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def __getitem__(self, item):
        pcd_path = self.data[item]

        # render_img_path = random.choice(get_render_imgs(pcd_path))
        render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)

        render_img_list = []
        for indexing in range(0,1):
            render_img_path = render_img_path_list[indexing]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform(render_img_a)
            render_img_list.append(render_img_a)

        for indexing1 in range(1,3):
            render_img_path = render_img_path_list[indexing1]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform1(render_img_a)
            render_img_list.append(render_img_a)

        # render_img = Image.open(render_img_path).convert('RGB')
        # render_img = self.transform(render_img)  #.permute(1, 2, 0)
        # render_img_list.append(render_img)

        pointcloud_1 = load_ply(self.data[item])
        pointcloud_orig = pointcloud_1.copy()

        point_t1 = trans_4(pointcloud_orig)
        
#         1024
        # pointcloud_orig1 = d_utils.fps(torch.from_numpy(pointcloud_orig).float(),1024)
        pointcloud_orig1 = self.farthest_point_sampling(pointcloud_orig)
        point_t2 = trans_3(pointcloud_orig1)
        
        
#         512
        # point_t3 = d_utils.fps(torch.from_numpy(pointcloud_orig).float(),512)
        point_t3 = self.farthest_point_sampling1(pointcloud_orig1)
        
        # pointcloud = (pointcloud_orig, point_t1, point_t2)
        # pointcloud = (point_t1, point_t2)
        # pointcloud=pointcloud_1

        # imgs = (render_img_list[0], render_img_list[1], render_img_list[2],render_img_list[3])
        imgs = (render_img_list[0], render_img_list[1], render_img_list[2])

        return (point_t1, point_t2,point_t3), imgs  # render_img # render_img_list

    def __len__(self):
        return len(self.data)


# 0130
class ShapeNetRender6(Dataset):
    def __init__(self, img_transform=None,img_transform1=None, n_imgs=1):
        self.data = load_shapenet_data()
        self.transform = img_transform
        self.transform1 = img_transform1
        self.n_imgs = n_imgs
    
    def farthest_point_sampling(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 1024:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def farthest_point_sampling1(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 512:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def __getitem__(self, item):
        pcd_path = self.data[item]

        # render_img_path = random.choice(get_render_imgs(pcd_path))
        render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)

        render_img_list = []
        for indexing in range(0,1):
            render_img_path = render_img_path_list[indexing]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform(render_img_a)
            render_img_list.append(render_img_a)

        for indexing1 in range(1,3):
            render_img_path = render_img_path_list[indexing1]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform1(render_img_a)
            render_img_list.append(render_img_a)

        pointcloud_1 = load_ply(self.data[item])
        pointcloud_orig = pointcloud_1.copy()

        
        point_t1 = trans_5(pointcloud_orig)
        
#         1024
       
        pointcloud_orig1 = self.farthest_point_sampling(pointcloud_orig)
        point_t2 = trans_3(pointcloud_orig1)
        
        
#         512个点
        
        point_t3 = self.farthest_point_sampling1(pointcloud_orig1)
        
        
        imgs = (render_img_list[0], render_img_list[1], render_img_list[2])

        return (point_t1, point_t2,point_t3), imgs  # render_img # render_img_list

    def __len__(self):
        return len(self.data)

class ShapeNetRender1(Dataset):
    def __init__(self, img_transform=None,img_transform1=None, n_imgs=1):
        self.data = load_shapenet_data()
        self.transform = img_transform
        self.transform1 = img_transform1
        self.n_imgs = n_imgs
    
    def farthest_point_sampling(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 1024:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def farthest_point_sampling1(self, point_cloud):
        num_points_orig = point_cloud.shape[0]
        sampled_indices = [np.random.randint(num_points_orig)]
        distances = cdist(point_cloud, point_cloud[sampled_indices])

        while len(sampled_indices) < 512:
        # while len(sampled_indices) < 2048:
            min_distances = np.min(distances, axis=1)
            new_index = np.argmax(min_distances)
            sampled_indices.append(new_index)
            new_distances = cdist(point_cloud, point_cloud[new_index].reshape(1, -1))
            distances = np.minimum(distances, new_distances)

        return point_cloud[sampled_indices]
    
    def __getitem__(self, item):
        pcd_path = self.data[item]

        # render_img_path = random.choice(get_render_imgs(pcd_path))
        render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)

        render_img_list = []
        for indexing in range(0,1):
            render_img_path = render_img_path_list[indexing]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform(render_img_a)
            render_img_list.append(render_img_a)

        for indexing1 in range(1,3):
            render_img_path = render_img_path_list[indexing1]
            render_img_a = Image.open(render_img_path).convert('RGB')
            render_img_a = self.transform1(render_img_a)
            render_img_list.append(render_img_a)

        # render_img = Image.open(render_img_path).convert('RGB')
        # render_img = self.transform(render_img)  #.permute(1, 2, 0)
        # render_img_list.append(render_img)

        pointcloud_1 = load_ply(self.data[item])
        pointcloud_orig = pointcloud_1.copy()

        point_t1 = trans_2(pointcloud_orig)
        
#         1024
        # pointcloud_orig1 = d_utils.fps(torch.from_numpy(pointcloud_orig).float(),1024)
        pointcloud_orig1 = self.farthest_point_sampling(pointcloud_orig)
        point_t2 = trans_3(pointcloud_orig1)
        
        
#         512
        # point_t3 = d_utils.fps(torch.from_numpy(pointcloud_orig).float(),512)
        point_t3 = self.farthest_point_sampling1(pointcloud_orig1)
     
        imgs = (render_img_list[0], render_img_list[1], render_img_list[2])

        return (point_t1, point_t2,point_t3), imgs  # render_img # render_img_list

    def __len__(self):
        return len(self.data)
    
def allDataset(transform, n_imgs):
    dataset = ShapeNetRender(transform, n_imgs)
    lenth = dataset.__len__()
    train_loader = DataLoader(dataset, batch_size=lenth, shuffle=True,num_workers=8, pin_memory=False,drop_last=False)
    dataiter = iter(train_loader)
    
    (point_t1, pointcloud_orig), imgs = dataiter.next()
 
    return (point_t1, pointcloud_orig), imgs
    

class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]



class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
        
        
def load_data_semseg(partition, test_area, train_area):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    root = 'data/'
    # download_S3DIS()
    # prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
        
    with open(os.path.join(root, 'indoor3d_sem_seg_hdf5_data/all_files.txt')) as f:
        all_files = [line.rstrip() for line in f]
        
    with open(os.path.join(root, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt')) as f:
        room_filelist = [line.rstrip() for line in f]
        
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
        
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            for area in train_area:
                if "Area_" + area in room_name:
                    train_idxs.append(i)
                    break
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg



class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1', train_area=['1','2','3','4','5','6']):
        self.data, self.seg = load_data_semseg(partition, test_area, train_area)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]