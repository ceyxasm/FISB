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
    # Save this image
    plt.savefig('../output/keypoints.png')
    plt.close()

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'SIFT' or method == 'AKAZE':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'ORB' or method == 'BRISK':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def showFeatureMatch(img1, img2, kpsA, kpsB, featuresA, featuresB, method, match):
    fig = plt.figure(figsize=(20,8))

    if match == 'BF':
        matches = matchKeyPointsBF(featuresA, featuresB, method=method)
        img3 = cv2.drawMatches(img1,kpsA,img2,kpsB,matches[:100],
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif match == 'Flann':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=method)
        img3 = cv2.drawMatches(img1,kpsA,img2,kpsB,np.random.choice(matches,100),
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('../output/matches.jpg', img3)
    return matches