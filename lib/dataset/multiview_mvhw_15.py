# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import collections
import json
import glob
import logging
import cv2

from dataset.joints_dataset import JointsDataset


logger = logging.getLogger(__name__)


origin_joints = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
    17: 'neck',  # fake
    18: 'root',  # fake, mid_hip
}

# # union(20) to h36m(15) / mvhw15
# UNION_TO_H36M15 = [0, 1, 2, 3, 4, 5, 6, 9, 11, 14, 15, 16, 17, 18, 19]

# fake(19) to h36m(15)
FAKE_TO_H36M15 = [18, 12, 14, 16, 11, 13, 15, 17, 0, 5, 7, 9, 6, 8, 10]


# rotate our 3d joints and R simultaneously， due to in the origin anno, our person heads down
M = np.array([[1.0, 0.0, 0.0],          # we need to rotate 180' by x axis
              [0.0, 1.0, 0.0],
              [0.0, 0.0, -1.0]])

with_dot_M = True

use_2d_gt = True

# used to filter unreasonable pose
reproject_thresh = None   # unit: pixel


def projectPoints(X, K, R, t, Kd):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.dot(R, X) + t

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                            r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                            r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


class MultiViewMVHW(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0:'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'neck',
            8: 'nose',
            9: 'lsho',
            10: 'lelb',
            11: 'lwri',
            12: 'rsho',
            13: 'relb',
            14: 'rwri'
        }

        if os.path.isdir('/home/yandanqi'):
            self.root = '/home/yandanqi/0_data/ourdata'
        elif os.path.isdir('/mntnfs/med_data5/yantianshuo/'):
            self.root = '/mntnfs/med_data5/yantianshuo/ourdata'
        else:
            self.root = '/data/posedata/mvhw'
        print(f"current dataset root = {self.root}")
        if self.root == '/data/posedata/mvhw':
            self.image_dir = os.path.join(self.root, f'{image_set}_images_oddviews')
        else:
            self.image_dir = os.path.join(self.root, 'images_oddviews')
        print(f"current image dir = {self.image_dir}")
        # self.image_dir = os.path.join(self.dataset_root, 'images_oddviews')
        self.data_format = 'images'
        self.image_set = image_set

        if os.path.isfile(os.path.join(self.root, 'ignore_list.txt')):
            self.ignore_list = self._sessionfile_to_list(os.path.join(self.root, 'ignore_list.txt'))
        else:
            self.ignore_list = list()

        if self.image_set == 'train':
            self.sequence_list = self._sessionfile_to_list(os.path.join(self.root, 'train_list.txt'))[:-3]
            # self.sequence_list = self._sessionfile_to_list(os.path.join(self.dataset_root, 'train_list.txt'))
            self._interval = 1
        elif self.image_set == 'validation':
            # self.sequence_list = self._sessionfile_to_list(os.path.join(self.dataset_root, 'valid_list.txt'))
            self.sequence_list = self._sessionfile_to_list(os.path.join(self.root, 'train_list.txt'))[-3:]
            self._interval = 1
            # self.cam_list = ['c01', 'c03', 'c05', 'c07']
        elif self.image_set == 'valid':
            # self.sequence_list = self._sessionfile_to_list(os.path.join(self.dataset_root, 'valid_list.txt'))
            self.sequence_list = self._sessionfile_to_list(os.path.join(self.root, 'valid_list.txt'))
            self._interval = 1
            # self.cam_list = ['c01', 'c03', 'c05', 'c07']
        elif self.image_set == 'test':
            # self.sequence_list = self._sessionfile_to_list(os.path.join(self.dataset_root, 'valid_list.txt'))
            self.sequence_list = self._sessionfile_to_list(os.path.join(self.root, 'test_list.txt'))
            self._interval = 1
            # self.cam_list = ['c01', 'c03', 'c05', 'c07']

        # if self.use_even_views:
        #     self.cam_list = ['c02', 'c04', 'c06', 'c08']
        # else:
        #     self.cam_list = ['c01', 'c03', 'c05', 'c07']
        self.even_views = cfg.DATASET.EVEN_VIEWS
        if self.even_views:
            self.cam_list = ['c02', 'c04', 'c06', 'c08']
        else:
            self.cam_list = ['c01', 'c03', 'c05', 'c07']
        # self.cam_list = ['c01', 'c03', 'c05', 'c07']
        self.num_views = len(self.cam_list)
        print(f"cam_list = {self.cam_list}")

        # self.db_file = 'group_{}_cam{}_tp_15.pkl'.format(self.image_set, self.num_views)
        # self.db_file = os.path.join(self.root, self.db_file)
        #
        # if osp.exists(self.db_file):
        #     info = pickle.load(open(self.db_file, 'rb'))
        #     assert info['sequence_list'] == self.sequence_list
        #     assert info['interval'] == self._interval
        #     assert info['cam_list'] == self.cam_list
        #     self.db = info['db']
        # else:
        #     self.db = self._get_db()
        #     info = {
        #         'sequence_list': self.sequence_list,
        #         'interval': self._interval,
        #         'cam_list': self.cam_list,
        #         'db': self.db
        #     }
        #     pickle.dump(info, open(self.db_file, 'wb'))
        self.db = self._get_db()
        self.db_size = len(self.db)

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)

        print(f"{image_set} totally has {self.db_size} items...")

    def _sessionfile_to_list(self, filepath):
        with open(filepath, 'r') as fr:
            lines = fr.readlines()
        return [item.strip() for item in lines]

    def get_group(self, db):
        """
        :return: [[0,1,2,3], [4,5,6,7], ...]
        """
        num_all = (len(db)//self.num_views)*self.num_views
        grouping = np.array([i for i in range(num_all)])

        return grouping.reshape(-1, self.num_views)

    def _get_fake_points(self, pose):

        """
        17->19->15
        :param pose: 17*2 or 17*3
        :return:
        """
        assert pose.shape[0] == 17 and pose.shape[1] in [2, 3]

        # generate neck by average left_shoulder and right_should
        pose_neck = (pose[5] + pose[6])/2.0
        # generate hip-mid(root) by average left_hip and right_hip
        pose_root = (pose[11] + pose[12])/2.0

        pose = np.append(pose, [pose_neck, pose_root], axis=0)   # [19, 3]

        return pose[FAKE_TO_H36M15]

    def _get_db(self):
        width = 1080
        height = 1920
        db = []
        for seq in self.sequence_list:
            if seq in self.ignore_list:
                continue

            # get camera anno in this seq
            cameras = self._get_cam(seq)

            # get anno file for this seq
            kpts_2d_file = os.path.join(self.root, 'keypoints2d', f'{seq}.npy')
            kpts_3d_file = os.path.join(self.root, 'keypoints3d_anioptim', f'{seq}.npy')
            kpts_2d = np.load(kpts_2d_file, allow_pickle=True)[0][
                'keypoints2d']  # dict_keys(['keypoints2d', 'center', 'scale', 'keypoints2d_repro'])
            kpts_3d = np.load(kpts_3d_file, allow_pickle=True)[0][
                'keypoints3d_optim']  # dict_keys(['keypoints3d', 'keypoints3d_optim', 'keypoints3d_smoothnet', 'name', 'nframes'])

            # get bbx params
            bbx_all = self._get_bbx_params(seq)

            # get other info
            subject_id = seq.split('_')[-1][-2:]

            # get image list
            image_list = sorted(glob.glob(os.path.join(self.image_dir, f'{seq}_c01', '*.jpg')))

            # traverse frames
            frame_len = len(image_list)
            if frame_len > 1:
                for idx in range(frame_len):
                    image_name = os.path.basename(image_list[idx])
                    _, _, _, _, frame_idx = image_name.split('.')[0].split('_')
                    frame_idx = int(frame_idx)

                    # traverse cameras
                    paired_group = []
                    is_paired = True
                    for cam_name, cam in cameras.items():
                        cam_id = int(cam_name[-1])

                        # get image path
                        img_path = os.path.join(self.image_dir, f'{seq}_{cam_name}',
                                                f'{seq}_{cam_name}_{frame_idx:06d}.jpg')

                        # get pose
                        pose_3d = self._get_fake_points(kpts_3d[int(frame_idx)]) # [num_joints, 3]
                        if with_dot_M:
                            pose_3d = pose_3d.dot(M)
                        if use_2d_gt:
                            pose_2d = self._get_fake_points(kpts_2d[cam_id - 1, int(frame_idx), :, :2])  # [num_joints, 3]
                        else:
                            pose_2d = projectPoints(X=pose_3d.T, K=cam['K'], R=cam['R'], t=cam['t'], Kd=cam['distCoef']).T[:,
                                      :2]  # T -- cm

                        poses_3d = pose_3d*10   # cm to mm
                        poses_vis = np.ones_like(pose_3d)
                        poses = pose_2d

                        # check if the pose_3d is consistence with pose_2d by given cam params
                        pose_2d_proj = projectPoints(X=pose_3d.T, K=cam['K'], R=cam['R'], t=cam['t'], Kd=cam['distCoef']).T[:, :2]  # T -- cm
                        diff_max = np.abs(pose_2d - pose_2d_proj).max()
                        if reproject_thresh is not None:
                            if diff_max > reproject_thresh:
                                is_paired = False
                                print(f"reproject pose unreasonable! {seq} - {frame_idx:06d}")
                                break

                        # get cam
                        our_cam = dict()
                        our_cam['R'] = cam['R']
                        our_cam['T'] = -np.dot(cam['R'].T, cam['t']) * 10  # cm to mm
                        our_cam['fx'] = np.array([cam['K'][0, 0]])
                        our_cam['fy'] = np.array([cam['K'][1, 1]])
                        our_cam['cx'] = np.array([cam['K'][0, 2]])
                        our_cam['cy'] = np.array([cam['K'][1, 2]])
                        our_cam['k'] = cam['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = cam['distCoef'][[2, 3]].reshape(2, 1)
                        # print(f"our_cam['fx'] = {our_cam['fx']}")

                        # get center and scale for crop
                        bbx = bbx_all[cam_id-1, frame_idx]  # (5, ) --- [x, y, h, w]
                        center = (bbx[0] + 0.5*bbx[2], bbx[1] + 0.5*bbx[3])
                        scale = (bbx[2]/100, bbx[3]/100.0)

                        # db.append({
                        #     'source': 'mvhw',
                        #     'key': "{}_{}-{}".format(seq, cam_name, str(frame_idx).zfill(6)),
                        #     'subject': subject_id,
                        #     'camera_id': cam_name,
                        #     'image': img_path,
                        #     'image_name': os.path.basename(img_path),
                        #     'joints_3d': poses_3d,
                        #     'joints_2d': poses,
                        #     'joints_vis': poses_vis,
                        #     'camera': our_cam,
                        #     'center': center,
                        #     'scale': scale,
                        # })
                        paired_group.append({
                            'source': 'mvhw',
                            'key': "{}_{}-{}".format(seq, cam_name, str(frame_idx).zfill(6)),
                            'subject': subject_id,
                            'camera_id': cam_name,
                            'image': img_path,
                            'image_name': os.path.basename(img_path),
                            'joints_3d': poses_3d,
                            'joints_2d': poses,
                            'joints_vis': poses_vis,
                            'camera': our_cam,
                            'center': center,
                            'scale': scale
                        })
                    if is_paired:
                        for db_rec in paired_group:
                            db.append(db_rec)

        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.root, 'cameras', f'{seq}.json')
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        cameras = {}
        for cam in calib:
            if cam['name'] in self.cam_list:  # 'c01', 'c03', 'c05', 'c07' cams if num_list=4
                # cam --- dict_keys(['name', 'size', 'matrix', 'rotation', 'distortions', 'translation'])
                sel_cam = dict()
                sel_cam['K'] = np.array(cam['matrix'])
                sel_cam['distCoef'] = np.array(cam['distortions'])
                sel_cam['R'] = cv2.Rodrigues(np.array(cam['rotation']))[0]
                if with_dot_M:
                    sel_cam['R'] = sel_cam['R'].dot(M)
                sel_cam['t'] = np.array(cam['translation']).reshape((3, 1))  # cm
                cameras[cam['name']] = sel_cam
        return cameras

    def _get_bbx_params(self, seq):
        bbx_file = osp.join(self.root, 'bbox2d', f'{seq}.npy')
        bbox_all = np.load(bbx_file)    # [8, 350, 5]
        return bbox_all

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        for k in range(self.num_views):
            i, t, w, m = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, pred, *args, **kwargs):
        gt_num = self.db_size
        assert len(pred) == gt_num, f'{len(pred)}pred, {gt_num} gt_num, number mismatch'

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
        # for i in range(gt_num):
        #     # index = self.num_views * i
        #     # print(f"i = {i}")
        #     # print(f"su = {su}")
        #     # print(f"shape of self.db[i]['joints_2d'] = {self.db[i]['joints_2d'].shape}")
        #     gt.append(self.db[i]['joints_2d'][su, :2])
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


