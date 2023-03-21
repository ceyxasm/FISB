from lib import *

def getHomography(kpsA, kpsB, matches, thresh):
    if len(matches) > 4:
        src_pts = np.float32([ kpsA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([ kpsB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        # estimate the homography between the sets of points
        (M, mask) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, thresh)
        mask = mask.ravel().tolist()
        return (M, mask)
    else:
        return None
