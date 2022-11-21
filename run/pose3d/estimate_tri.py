# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import pickle
import json
import argparse
import numpy as np

import _init_paths
from core.inference import get_max_preds
from core.config import config
from core.config import update_config
from utils.utils import create_logger
from multiviews.pictorial import rpsm
from multiviews.body import HumanBody
from multiviews.cameras import camera_to_world_frame
from multiviews.triangulate import triangulate_poses
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D Pose Estimation')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def compute_limb_length(body, pose):
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx] - pose[child])
            limb_length[(idx, child)] = length
    return limb_length


def _calculate_mpjpe_valid(mpjpes, threshold=500):
    mpjpe_all = []
    cnt = 0
    for mpjpe in mpjpes:
        if mpjpe < threshold:
            mpjpe_all.append(mpjpe)
            cnt += 1
    valid_ratio = 100 * (cnt/len(mpjpes))
    mpjpe_mean = np.array(mpjpe_all).mean()
    return mpjpe_mean, valid_ratio


def _eval_list_to_ap(mpjpes, threshold):
    total_num = len(mpjpes)

    tp = np.zeros(total_num)
    fp = np.zeros(total_num)
    for i, mpjpe in enumerate(mpjpes):
        if mpjpe < threshold:
            tp[i] = 1
        else:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (total_num + 1e-5)
    precise = tp / (tp + fp + 1e-5)
    for n in range(total_num - 2, -1, -1):
        precise[n] = max(precise[n], precise[n + 1])

    precise = np.concatenate(([0], precise, [0]))
    recall = np.concatenate(([0], recall, [1]))
    index = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

    return ap, recall[-2]


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test3d-tri')

    prediction_path = os.path.join(final_output_dir,
                                   config.TEST.HEATMAP_LOCATION_FILE)
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)

    all_locations = h5py.File(prediction_path)['locations']     # (#sample, 17, 2)
    all_heatmaps = h5py.File(prediction_path)['heatmaps']       # (#sample, 17, 64, 64)
    print(f"load prediction results from{prediction_path}")

    cnt = 0
    grouping = test_dataset.grouping
    mpjpes = []
    mpjpe_score_output = {}
    pose3d_output = {}

    for items in grouping:      # all 4 views for one subject
        heatmaps = []
        locations = []
        poses = []          # save same 3D world coordinate 4 times
        cameras = []

        # get information of all 4 views
        for idx in items:   #
            datum = test_dataset.db[idx]
            camera = datum['camera']        # dict: {R, T, fx, fy, cx, cy}
            cameras.append(camera)

            if 'h36m' in config.DATASET.TEST_DATASET:
                poses.append(camera_to_world_frame(datum['joints_3d_camera'], camera['R'],
                                               camera['T']))       # pose in 3D world coordinate (17, 3)
            elif 'mvhw' in config.DATASET.TEST_DATASET:
                poses.append(datum['joints_3d'])    # pose in 3d world
            else:
                raise Exception(f"pose not define in {config.DATASET.TEST_DATASET}")

            locations.append(all_locations[cnt])
            heatmaps.append(all_heatmaps[cnt])
            cnt += 1

        # s_11_act_16_subact_01_ca_04/s_11_act_16_subact_01_ca_04_000090.jpg
        keypoint_vis = datum['joints_vis']  # (20, 3)
        u2a = test_dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        u = np.array(list(a2u.values()))
        keypoint_vis = keypoint_vis[u]          # (17, 3)

        locations = np.array(locations)[:, :, :2]               # (#view, 17, 2) in original scale
        heatmaps = np.array(heatmaps)                           # (#view, 17, 64, 64)
        _, confs = get_max_preds(heatmaps)                      # (#view, 17, 1)

        prediction = triangulate_poses(cameras, locations, confs.squeeze())      # list, element: (17, 3)

        mpjpe = np.mean(np.sqrt(np.sum((prediction[0] * keypoint_vis - poses[0] * keypoint_vis)**2, axis=1)))
        mpjpes.append(mpjpe)

        print(mpjpe)
        if mpjpe > 150:
            print('Wrong MPJPE !!! ', datum['image'])

        # ================== save MPJPE score ==================
        if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
            datum['image'] = datum['image'].replace('\\', '/')
            seq, frame = datum['image'].split('/')[1].split('_ca_')      # s_11_act_16_subact_01, 04_000090.jpg
            frame_name = seq + frame[2:-4]
            mpjpe_score_output[frame_name] = mpjpe

            # ================== save 3D pose ================
            pose3d_output[frame_name] = {}
            pose3d_output[frame_name]['pred'] = prediction[0].tolist()      # from numpy to list
            pose3d_output[frame_name]['GT'] = poses[0].tolist()

    # calculate ap
    mpjpe_threshold = np.arange(25, 155, 25)
    aps = []
    recs = []
    for t in mpjpe_threshold:
        ap, rec = _eval_list_to_ap(mpjpes=mpjpes, threshold=t)
        aps.append(ap)
        recs.append(rec)
    msg = f'ap@25: {aps[0]:.4f}\tap@50: {aps[1]:.4f}\tap@75: {aps[2]:.4f}\t \
          ap@100: {aps[3]:.4f}\tap@125: {aps[4]:.4f}\tap@150: {aps[5]:.4f}\t \
          recall@500mm: {recs[-2]:.4f}\t'
    logger.info(msg)

    logger.info('Triangulation MPJPE {}'.format(np.mean(mpjpes)))
    mpjpes_valid, valid_ratio = _calculate_mpjpe_valid(mpjpes)
    logger.info(f'Valid Triangulation MPJPE: {mpjpes_valid}, valid_ratio: {valid_ratio}')

    # if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
    json.dump(mpjpe_score_output, open(os.path.join(final_output_dir, 'mpjpe_score.json'), 'w'), indent=4, sort_keys=True)
    json.dump(pose3d_output, open(os.path.join(final_output_dir, 'output_3d_joint.json'), 'w'), indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
