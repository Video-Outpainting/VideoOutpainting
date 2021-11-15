# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:39:54 2018

@author: carri
"""
# for testing case
from __future__ import division

import os
import numpy as np
import cv2
import random

from torch.utils.data import Dataset

class PairwiseImg(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.range = sample_range
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        img_list = []
        Index = {}
        
        seq = self.seq_name
        
        images = np.sort(os.listdir(db_root_dir))
        images_path = images #list(map(lambda x: os.path.join('blackswan'.strip(), x), images))

        start_num = len(img_list)
        img_list.extend(images_path)
        end_num = len(img_list)
        Index['']= np.array([start_num, end_num])

        """
        images = np.sort(os.listdir(db_root_dir))
        print(db_root_dir,images)
        images_path = list(map(lambda x: x, images))
        print(db_root_dir,images)
        print(images_path)
        start_num = len(img_list)
        img_list.extend(images)
        end_num = len(img_list)
        Index[seq.strip('\n')]= np.array([start_num, end_num])"""

        
        
        self.img_list = img_list
        self.Index = Index


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        target, target_gt,sequence_name = self.make_img_gt_pair(idx) 
        target_id = idx
        seq_name1 = self.img_list[target_id].split('/')[-1] 
        sample = {'target': target, 'target_gt': target_gt, 'seq_name': sequence_name, 'search_0': None}
  
        my_index = self.Index['']
        search_num = list(range(my_index[0], my_index[1]))  
        search_ids = random.sample(search_num, self.range)
        print(search_ids)
        for i in range(0,self.range):
            search_id = search_ids[i]
            search, search_gt,sequence_name = self.make_img_gt_pair(search_id)
            if sample['search_0'] is None:
                sample['search_0'] = search
            else:
                sample['search'+'_'+str(i)] = search

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname
       


        return sample
    
    def make_img_gt_pair(self, idx): 
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]), cv2.IMREAD_COLOR)
        gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
        img = np.array(img, dtype=np.float32)

        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))        
        img = img.transpose((2, 0, 1))  
        
        
        sequence_name = self.img_list[idx].split('/')[-1]
        return img, gt, sequence_name 


