import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy
import cv2
# from torch.utils.data import Dataset

from dataset.JointsDataset import JointsDataset
from utils import utils
from utils import cameras as util_cams


logger = logging.getLogger(__name__)


# TRAIN_LIST = ['20220618_1b492fd601_subj22', '20220618_31a9f57701_subj22']
# TRAIN_LIST = ['20220618_31a9f57701_subj22']
# VAL_LIST = ['20220619_0ce09a6b01_subj19']

use_2d_gt = True


JOINTS_DEF = {
    'nose': 0,
    'l-eye': 1,
    'r-eye': 2,
    'l-ear': 3,
    'r-ear': 4,
    'l-shoulder': 5,
    'r-shoulder': 6,
    'l-elbow': 7,
    'r-elbow': 8,
    'l-wrist': 9,
    'r-wrist': 10,
    'l-hip': 11,
    'r-hip': 12,
    'l-knee': 13,
    'r-knee': 14,
    'l-ankle': 15,
    'r-ankle': 16,
}

LIMBS = [[0, 1],
         [0, 2],
         [1, 3],
         [2, 4],
         [1, 5],
         [2, 6],
         [5, 7],
         [8, 6],
         [9, 7],
         [10, 8],
         [11, 5],
         [12, 11],
         [13, 11],
         [14, 12],
         [15, 13],
         [14, 16]]


def generate_fake_points(idx):
    # define fake points
    num_views = 5
    num_points = len(JOINTS_DEF)

    f_path = os.path.join('data', 'mvhw_chessboard_ori.pkl')
    f = utils.load_pickle_file(f_path)
    points_2d = np.array(f['p2ds'])[:num_views, :num_points, :]   # [num_views, num_points, 2],

    points_3d = np.array(f['p3ds'])[:num_points, :] * 10   # [num_points, 3], cm to mm

    return points_2d[idx], points_3d


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


class MVHW_Image(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.dataset_root = cfg.DATASET.ROOT        # '/home/yandanqi/0_data/ourdata'     # delete this line in the future
        self.image_dir = os.path.join(self.dataset_root, 'images_oddviews')
        self.data_format = 'images'
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = 17    # len(JOINTS_DEF)
        if os.path.isfile(os.path.join(self.dataset_root, 'ignore_list.txt')):
            self.ignore_list = self._sessionfile_to_list(os.path.join(self.dataset_root, 'ignore_list.txt'))
        else:
            self.ignore_list = list()
        if self.image_set == 'train':
            self.sequence_list = self._sessionfile_to_list(os.path.join(self.dataset_root, 'train_list.txt'))
            # self.cam_list = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08']
            self.cam_list = ['c01', 'c03', 'c05', 'c07']
        elif self.image_set == 'validation':
            self.sequence_list = self._sessionfile_to_list(os.path.join(self.dataset_root, 'valid_list.txt'))
            self.cam_list = ['c01', 'c03', 'c05', 'c07']
        self.num_views = len(self.cam_list)

        self.db = self._get_db()
        self.db_size = len(self.db)

    def _sessionfile_to_list(self, filepath):
        with open(filepath, 'r') as fr:
            lines = fr.readlines()
        return [item.strip() for item in lines]

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
            kpts_2d_file = os.path.join(self.dataset_root, 'keypoints2d', f'{seq}.npy')
            kpts_3d_file = os.path.join(self.dataset_root, 'keypoints3d_anioptim', f'{seq}.npy')
            kpts_2d = np.load(kpts_2d_file, allow_pickle=True)[0]['keypoints2d']       # dict_keys(['keypoints2d', 'center', 'scale', 'keypoints2d_repro'])
            kpts_3d = np.load(kpts_3d_file, allow_pickle=True)[0]['keypoints3d_optim']       # dict_keys(['keypoints3d', 'keypoints3d_optim', 'keypoints3d_smoothnet', 'name', 'nframes'])

            image_list = sorted(glob.glob(os.path.join(self.image_dir, f'{seq}_c01', '*.jpg')))
            frame_len = len(image_list)
            # print(seq, seq_len, image_list[-1], kpts_2d.shape, kpts_3d.shape)
            # traverse frames
            for idx in range(frame_len):
                img_numpy = image_list[idx]
                image_name = os.path.basename(img_numpy)
                _, _, _, cam_name, frame_idx = image_name.split('.')[0].split('_')
                frame_idx = int(frame_idx)
                if np.isnan(kpts_2d[::2, frame_idx]).any() or np.isnan(kpts_3d[frame_idx]).any():
                    logger.info(f'NaN found in keypoints! {seq} - {frame_idx:06d}')
                    continue

                # traverse cameras
                for cam_name in cameras:
                    img_path = os.path.join(self.image_dir, f'{seq}_{cam_name}', f'{seq}_{cam_name}_{frame_idx:06d}.jpg')

                    cam_id = int(cam_name[-1])
                    vv = cameras[cam_name]
                    
                    # get pose
                    all_poses_3d = []
                    all_poses_vis_3d = []
                    all_poses = []
                    all_poses_vis = []

                    pose_3d = kpts_3d[int(frame_idx)]                           # [num_joints, 3]
                    if use_2d_gt:
                        pose_2d = kpts_2d[cam_id - 1, int(frame_idx), :, :2]    # [num_joints, 3]
                    else:
                        pose_2d = projectPoints(X=pose_3d.T, K=vv['K'], R=vv['R'], t=vv['t'], Kd=vv['distCoef']).T[:, :2]  # T -- cm

                    all_poses_3d.append(pose_3d * 10)     # cm to mm
                    all_poses_vis_3d.append(np.ones_like(pose_3d))
                    all_poses.append(pose_2d)
                    all_poses_vis.append(np.ones_like(pose_2d))

                    our_cam = dict()
                    our_cam['R'] = vv['R']
                    our_cam['T'] = -np.dot(vv['R'].T, vv['t']) * 10  # cm to mm
                    our_cam['fx'] = np.array(vv['K'][0, 0])
                    our_cam['fy'] = np.array(vv['K'][1, 1])
                    our_cam['cx'] = np.array(vv['K'][0, 2])
                    our_cam['cy'] = np.array(vv['K'][1, 2])
                    our_cam['k'] = vv['distCoef'][[0, 1, 4]].reshape(3, 1)
                    our_cam['p'] = vv['distCoef'][[2, 3]].reshape(2, 1)

                    db.append({
                        'key': "{}_{}-{}".format(seq, cam_name, str(frame_idx).zfill(6)),
                        'image': img_path,
                        'image_name': os.path.basename(img_path),
                        'joints_3d': all_poses_3d,
                        'joints_3d_vis': all_poses_vis_3d,
                        'joints_2d': all_poses,
                        'joints_2d_vis': all_poses_vis,
                        'camera': our_cam
                    })
                
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, 'cameras',  f'{seq}.json')
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
                sel_cam['t'] = np.array(cam['translation']).reshape((3, 1))    # cm
                cameras[cam['name']] = sel_cam
        return cameras

    
    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        for k in range(self.num_views):         # get one view
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return aps, recs, self._eval_list_to_mpjpe(eval_list), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt





if __name__ == '__main__':
    pass

    print(f'kkkkkkkkkkkkkkkkkkkkk')
