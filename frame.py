import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform 

IRt = np.eye(4)

# turn [[x, y]] -> [[x, y, 1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

def extractRt(E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    if os.getenv('REVERSE'):
        t *= -1
    return np.linalg.inv(poseRt(R, t))


def extract(img):
    orb = cv2.ORB_create()

    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
    
    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(img, kps)

    # return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def normalize(Kinv, pts):
    # TODO: why is z=1? 
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2] # remove the ones

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    ret = []
    idx1, idx2 = [], []

    # Lowe's ratio test
    for m, n in matches:
        # TODO: find out what .distance means   
        if m.distance < 0.75*n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]
            
            # ensure matched points are not too far apart from each other (since they are consecutive frames in a video)
            if m.distance < 32:
                # avoid duplicates (TODO: not sure why there would be duplicates)
                if m.queryIdx not in idx1 and m.trainIdx not in idx2:
                    # keep around the indices
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                
                    ret.append((p1, p2))

    # no duplicates
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    assert len(ret) >= 8
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    ret = np.array(ret)

    # fit matrix and ignore outliers
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            #EssentialMatrixTransform,
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,
                            #residual_threshold=0.02,
                            max_trials=100)
    
    print(f'Matches: {len(f1.des)} -> {len(matches)} -> {len(inliers)} -> {sum(inliers)}')

    # extract pose from FundamentalMatrix
    Rt = extractRt(model.params)

    # return
    return idx1[inliers], idx2[inliers], Rt


class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = np.eye(4)
        self.h, self.w = img.shape[0:2]

        self.kpus, self.des = extract(img)
        self.kd = cKDTree(self.kpus)
        self.kps = normalize(self.Kinv, self.kpus)
        self.pts = [None]*len(self.kps)
        
        self.id = mapp.add_frame(self)

