#!/usr/bin/env python3

import os
import sys
import time
import cv2
import numpy as np
import warnings
import g2o

from frame import Frame, match_frames, denormalize, IRt
from pointmap import Map, Point
from display import Display

# camera intrinsics
# W, H = 1920//2, 1080//2
W, H = 1242, 375

F = int(os.getenv("F", "800"))
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]

        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret

    # return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

frames = []
def process_frame(img):
    img = cv2.resize(img, (W, H))
    
    frame = Frame(mapp, img, K)
    frames.append(frame)
    if frame.id == 0:
        return 

    print(f'\n*** frame {frame.id} ***')

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    # if there is already a match, exclude it from future
    for i, idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])

    # unmatched points (i.e., first time match)
    good_pts4d = np.array([f1.pts[i] is None for i in idx1])

    # locally in front of camera
    # reject pts without enough parallax
    pts_tri_local = triangulate(Rt, np.eye(4), f1.kps[idx1], f2.kps[idx2]) # idx1,, idx2 are the indices of the matched frames
    good_pts4d &= np.abs(pts_tri_local[:, 3]) > 0.005

    # homogeneous 3D coords 
    # reject points behind the camera
    pts_tri_local /= pts_tri_local[:, 3:]
    good_pts4d &= pts_tri_local[:, 2] > 0

    print(f'Adding {np.sum(good_pts4d)} points')

    # project into world
    pts4d = np.dot(np.linalg.inv(f1.pose), pts_tri_local.T).T

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue

        u,v = int(round(f1.kpus[idx1[i],0])), int(round(f1.kpus[idx1[i],1]))

        # add good points to respective frames
        pt = Point(mapp, p, img[v, u])
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        # denormalize points for display
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    # 2D display
    if disp is not None:
        disp.paint(img)

    # optimize the map
    if os.getenv("OPT") and frame.id >= 4:
        err = mapp.optimize()
        print(f'Optimize {err} units of error')

    # 3D map
    mapp.display()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]} <video.mp4>')
        exit(1)

    cap = cv2.VideoCapture(sys.argv[1])

    mapp = Map()
    if os.getenv("D3D") is not None:
        mapp.create_viewer()

    disp = None
    if os.getenv("D2D") is not None:
        disp = Display(W, H) 

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break

