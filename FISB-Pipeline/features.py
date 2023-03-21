from lib import *

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'akaze', 'brisk', 'orb'"
    
    # detect and extract features from the image
    if method == 'SIFT':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'AKAZE':
        descriptor = cv2.AKAZE_create()
    elif method == 'BRISK':
        descriptor = cv2.BRISK_create()
    elif method == 'ORB':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

def showKeypoints(img1, img2, kpsA, kpsB):
    # display the keypoints and features detected on both images
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(img1,kpsA,None,color=(0,255,0)))
    ax1.set_xlabel("(a)", fontsize=14)
    ax2.imshow(cv2.drawKeypoints(img2,kpsB,None,color=(0,255,0)))
    ax2.set_xlabel("(b)", fontsize=14)
    plt.savefig('../output/keypoints.png')
    plt.close()

def createMatcher(method,crossCheck):
    # Create and return a Matcher Object
    if method == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'ORB' or method == 'BRISK' or method == 'AKAZE':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
    best_matches = bf.match(featuresA,featuresB)
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def matchKeyPointsFlann(featuresA, featuresB, method, ratio):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    rawMatches = flann.knnMatch(featuresA,featuresB,k=2)
    print("Raw matches (flann):", len(rawMatches))
    matches = []
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def showFeatureMatch(img1, img2, kpsA, kpsB, featuresA, featuresB, method, match):

    if match == 'BF':
        matches = matchKeyPointsBF(featuresA, featuresB, method=method)
        img3 = cv2.drawMatches(img1,kpsA,img2,kpsB,matches[:100],
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif match == 'KNN':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=1, method=method)
        img3 = cv2.drawMatches(img1,kpsA,img2,kpsB,np.random.choice(matches,100),
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif match == 'FLANN':
        matches = matchKeyPointsFlann(featuresA, featuresB, method=method, ratio=1)
        img3 = cv2.drawMatches(img1,kpsA,img2,kpsB,np.random.choice(matches,100),
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite('../output/matches.jpg', img3)
    return matches