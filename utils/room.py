import math

import numpy as np 
import cv2


def compute_rel(box1, box2):
    center1 = np.array([(box1[0] + box1[3]) / 2, (box1[1] + box1[4]) / 2, (box1[2] + box1[5]) / 2])
    center2 = np.array([(box2[0] + box2[3]) / 2, (box2[1] + box2[4]) / 2, (box2[2] + box2[5]) / 2])

    # random relationship
    sx0, sy0, sz0, sx1, sy1, sz1 = box1
    ox0, oy0, oz0, ox1, oy1, oz1 = box2
    d = center1 - center2
    theta = math.atan2(d[2], d[0])  # range -pi to pi

    distance = (d[2]**2 + d[0]**2)**0.5
    
    # "on" relationship
    p = None
    if center1[0] >= box2[0] and center1[0] <= box2[3]:
        if center1[2] >= box2[2] and center1[2] <= box2[5]:
            delta1 = center1[1] - center2[1]
            delta2 = (box1[4] - box1[1] + box2[4] - box2[1]) / 2
            if 0 <(delta1 - delta2) < 0.05:
                p = 'on'
            elif 0.05 < (delta1 - delta2):
                p = 'above'
        return p, distance

    # eliminate relation in vertical axis now
    if abs(d[1]) > 0.5:
        return p, distance

    area_s = (sx1 - sx0) * (sz1 - sz0)
    area_o = (ox1 - ox0) * (oz1 - oz0)
    ix0, ix1 = max(sx0, ox0), min(sx1, ox1)
    iz0, iz1 = max(sz0, oz0), min(sz1, oz1)
    area_i = max(0, ix1 - ix0) * max(0, iz1 - iz0)
    iou = area_i / (area_s + area_o - area_i)
    touching = 0.0001 < iou < 0.5

    if sx0 < ox0 and sx1 > ox1 and sz0 < oz0 and sz1 > oz1:
        p = 'surrounding'
    elif sx0 > ox0 and sx1 < ox1 and sz0 > oz0 and sz1 < oz1:
        p = 'inside'
    # 60 degree intervals along each direction
    elif theta >= 5 * math.pi / 6 or theta <= -5 * math.pi / 6:
        p = 'right touching' if touching else 'left of'
    elif -2 * math.pi / 3 <= theta < -math.pi / 3:
        p = 'behind touching' if touching else 'behind'
    elif -math.pi / 6 <= theta < math.pi / 6:
        p = 'left touching' if touching else 'right of'
    elif math.pi / 3 <= theta < 2 * math.pi / 3:
        p = 'front touching' if touching else 'in front of'

    return p, distance

def get_corners(floor, threshold=0.1, viz=False):
    floor = floor.numpy()
    floor[floor >= 0.1] = 255
    floor[floor < 0.1] = 0
    thresh = floor.astype(np.uint8)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    corners = list(point[0] for point in contours[0])

    if viz:
        for (x, y) in corners:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    return corners, img