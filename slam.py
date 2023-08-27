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
from display import Display2D, Display3D


def hamming_distance(a, b):
  r = (1 << np.arange(8))[:,None]
  return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
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

def process_frame(img):
    start_time = time.time()
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return 


    f1 = mapp.frames[-1] # most recent frame
    f2 = mapp.frames[-2] # second to most recent frame

    # match keypoints between frames and obtain pose
    idx1, idx2, Rt = match_frames(f1, f2)

    if frame.id < 5:
        # get initial positions from fundamental matrix
        f1.pose = np.dot(Rt, f2.pose)
    else:
        # use kinematic model
        velocity = np.dot(f2.pose, np.linalg.inv(mapp.frames[-3].pose))
        f1.pose = np.dot(velocity, f2.pose)

    # if Point was already matched exclude it from future
    for i, idx in enumerate(idx2):
        # f2 - second most recent frame, point was added when it was most recent frame
        if f2.pts[idx] is not None:
            # add already matched points from previous frame to current frame (to be excluded later) 
            # to avoid mapping repeated points (?)
            f2.pts[idx].add_observation(f1, idx1[i])

    # pose optimization
    pose_opt = mapp.optimize(local_window=1, fix_points=True) # optimize latest frame

    # search by projection
    sbp_pts_count = 0
    if len(mapp.points) > 0:
        map_points = np.array([p.homogeneous() for p in mapp.points])
        projs = np.dot(np.dot(K, f1.pose[:3]), map_points.T).T
        projs = projs[:, 0:2] / projs[:, 2:]
        good_pts = (projs[:, 0] > 0) & (projs[:, 0] < W) & \
                   (projs[:, 1] > 0) & (projs[:, 1] < H)

        for i, p in enumerate(mapp.points):
            if not good_pts[i]:
                continue
            q = f1.kd.query_ball_point(projs[i], 5)
            for m_idx in q:
                if f1.pts[m_idx] is None:
                    # if any descriptors within 32
                    for o in p.orb():
                        o_dist = hamming_distance(o, f1.des[m_idx])
                        if o_dist < 32:
                            p.add_observation(f1, m_idx) # add already seen points
                            sbp_pts_count += 1
                            break

    # unmatched points (i.e., first time match), we added matched points above
    good_pts4d = np.array([f1.pts[i] is None for i in idx1])

    # reject pts without enough parallax
    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    good_pts4d &= np.abs(pts4d[:, 3]) > 0.005

    # homogeneous 3-D coords
    pts4d /= pts4d[:, 3:]

    # points locally in front of camera (BROKEN?)
    # pts_tri_local = np.dot(f1.pose, pts4d.T).T
    # good_pts4d &= pts_tri_local[:, 2] > 0

    print(f'Adding:  {np.sum(good_pts4d)}, new points, {sbp_pts_count} search by projection')

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue

        u,v = int(round(f1.kpus[idx1[i],0])), int(round(f1.kpus[idx1[i],1]))

        # add good points to respective frames
        pt = Point(mapp, p[0:3], img[v, u])
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for i1, i2 in zip(idx1, idx2):
        # denormalize points for display
        pt1 = f1.kps[i1]
        pt2 = f2.kps[i2]
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        if f1.pts[i1] is not None:
            if len(f1.pts[i1].frames) >= 5:
                cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
            else:
                cv2.circle(img, (u1, v1), color=(0, 128, 0), radius=3)
        else:
            # red means they don't match
            cv2.circle(img, (u1, v1), color=(0, 0, 0), radius=3)

        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    # 2D display
    if disp2d is not None:
        disp2d.paint(img)

    # optimize the map
    if os.getenv("OPT") and frame.id >= 4 and frame.id%5 == 0:
        err = mapp.optimize()
        print(f'Optimize {err} units of error')

    # 3D display
    if disp3d is not None:
        disp3d.paint(mapp)
    
    print(f"Map:     %d points, %d frames" % (len(mapp.points), len(mapp.frames)))
    print(f"Time:    {(time.time()-start_time)*1000.0}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]} <video.mp4>')
        exit(1)

   
    # display views
    mapp = Map()
    disp2d = None
    disp3d = None

    disp3d = Display3D()
    cap = cv2.VideoCapture(sys.argv[1])

    # camera parameters
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    F = float(os.getenv("F", "525"))
    if os.getenv("SEEK") is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(os.getenv("FRAME")))

    if W > 1024:
        downscale = 1024.0/W
        F *= downscale
        H = int(H * downscale)
        W = 1024

    # camera intrinsics
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    disp2d = Display2D(W, H) 

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        print(f'\n*** frame {i}/{CNT} ***')
        if ret == True:
            process_frame(frame)
        else:
            break
        i += 1

