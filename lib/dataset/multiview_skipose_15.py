# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# Revised by Haoyu Ma
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import os.path as osp
import numpy as np
import pickle
import collections

from dataset.joints_dataset import JointsDataset


SKI17_TO_H36M15 = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16]


class MultiViewSkiPose(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            # 7: 'belly',
            7: 'neck',
            8: 'nose',
            # 10: 'head',
            9: 'lsho',
            10: 'lelb',
            11: 'lwri',
            12: 'rsho',
            13: 'relb',
            14: 'rwri'
        }

        self.root = '/mntnfs/med_data5/junhaoran/dataset'
        if cfg.DATASET.CROP:
            anno_file = osp.join(self.root, 'skipose', 'annot',
                                 'ski_{}.pkl'.format(image_set))
            print(f"load anno for {image_set} set from {anno_file}")
        else:
            anno_file = osp.join(self.root, 'skipose', 'annot',
                                 'ski_{}.pkl'.format(image_set))
            print(f"load anno for {image_set} set from {anno_file}")

        self.db = self.load_db(anno_file)

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        print(f"skipose {image_set} totally has {self.group_size} group items")

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)

        # turn from 17 joints to 15 joints
        nitems = len(dataset)
        for i in range(nitems):
            dataset[i]['source'] = 'skipose_15'
            dataset[i]['joints_3d'] = dataset[i]['joints_3d'][SKI17_TO_H36M15]
            dataset[i]['joints_2d'] = dataset[i]['joints_2d'][SKI17_TO_H36M15]
            dataset[i]['joints_vis'] = dataset[i]['joints_vis'][SKI17_TO_H36M15]
            dataset[i]['image'] = os.path.join(self.root, 'skipose', 'images', dataset[i]['image'])

        return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1, -1, -1]     # 6 cameras
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        seq = datum['video_id']
        frame = datum['image_id']
        return 'seq_{:03d}_image_{:06d}'.format(seq, frame)

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))    # [ 0  1  2  3  4  5  6  7  9 11 12 14 15 16 17 18 19]

        gt = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])       # (17, 2) in original scale
        gt = np.array(gt)           # (num_sample, 17, 2) in original scale
        pred = pred[:, su, :2]      # (num_sample, 17, 2) in original scale

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        return name_values, np.mean(joint_detection_rate)
