from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import torch
import re
from torch.utils.data import Dataset
import networkx
import itertools


class DADDataset(Dataset):
    def __init__(self, data_path, feature, phase='training', toTensor=False, device=torch.device('cuda'), vis=False):
        self.data_path = os.path.join(data_path, feature + '_features')
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 100
        self.n_obj = 19
        self.fps = 20.0
        self.dim_feature = self.get_feature_dim(feature)

        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)

    def __len__(self):
        return len(self.files_list)

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s" % (filepath)
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['data']
            labels = data['labels']
            detections = data['det']

            if features.shape[0] == 10:
                features = features[0]
            if labels.shape[0] == 10:
                labels = labels[0]
            if detections.shape[0] == 10:
                detections = detections[0]
                
        except:
            raise IOError('Load data error! File: %s' % (data_file))
        if labels[1] > 0:
            toa = [90.0]
        else:
            toa = [self.n_frames + 1]

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)
            labels = torch.Tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            video_id = str(data['ID'])[5:11]
            return features, labels, toa, detections, video_id
        else:
            return features, labels, toa


class CrashDataset(Dataset):
    def __init__(self, data_path, feature, phase='train', toTensor=False, device=torch.device('cuda'), vis=False):
        self.data_path = data_path
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 50
        self.n_obj = 19
        self.fps = 10.0
        self.dim_feature = self.get_feature_dim(feature)
        self.files_list, self.labels_list = self.read_datalist(data_path, phase)
        self.toa_dict = self.get_toa_all(data_path)
        self.video_base_path = "/data/crash/videos"

    def __len__(self):
        return len(self.files_list)

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def read_datalist(self, data_path, phase):
        list_file = os.path.join(data_path, self.feature + '_features', f'{phase}.txt')
        assert os.path.exists(list_file), "file not exists: %s" % (list_file)
        fid = open(list_file, 'r')
        data_files, data_labels = [], []
        for line in fid.readlines():
            filename, label = line.rstrip().split(' ')
            data_files.append(filename)
            data_labels.append(int(label))
        fid.close()
        return data_files, data_labels

    def get_toa_all(self, data_path):
        toa_dict = {}
        annofile = os.path.join(data_path, 'videos', 'Crash-1500.txt')
        annoData = self.read_anno_file(annofile)
        for anno in annoData:
            labels = np.array(anno['label'], dtype=int)
            toa = np.where(labels == 1)[0][0]
            toa = min(max(1, toa), self.n_frames - 1)
            toa_dict[anno['vid']] = toa
        return toa_dict

    def find_video_file(self, vidname):
        crash_folder = os.path.join(self.video_base_path, 'Crash-1500')
        for video_file in os.listdir(crash_folder):
            if video_file.startswith(vidname):
                return os.path.join(crash_folder, video_file)

        normal_folder = os.path.join(self.video_base_path, 'Normal')
        for video_file in os.listdir(normal_folder):
            if video_file.startswith(vidname):
                return os.path.join(normal_folder, video_file)

        return None

    def read_anno_file(self, anno_file):
        assert os.path.exists(anno_file), "Annotation file does not exist! %s" % (anno_file)
        result = []
        with open(anno_file, 'r') as f:
            for line in f.readlines():
                items = {}
                items['vid'] = line.strip().split(',[')[0]
                labels = line.strip().split(',[')[1].split('],')[0]
                items['label'] = [int(val) for val in labels.split(',')]
                assert sum(items['label']) > 0, 'invalid accident annotation!'
                others = line.strip().split(',[')[1].split('],')[1].split(',')
                items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
                result.append(items)
        f.close()
        return result

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.feature + '_features', self.files_list[index])
        vidname = self.files_list[index].split('/')[-1].split('.')[0]
        assert os.path.exists(data_file), "file not exists: %s" % (data_file)

        video_path = self.find_video_file(vidname)
        if video_path is None:
            raise FileNotFoundError(f"Video file not found for {vidname} in 'Crash-1500' or 'Normal' folders")
        
        try:
            data = np.load(data_file)
            features = data['data']
            labels = data['labels']
            detections = data['det']
            vid = str(data['ID'])
        except:
            raise IOError('Load data error! File: %s' % (data_file))
        if labels[1] > 0:
            toa = [self.toa_dict[vid]]
        else:
            toa = [self.n_frames + 1]

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)
            labels = torch.Tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            return features, labels, toa, detections, vid, video_path
        else:
            return features, labels, toa, video_path
