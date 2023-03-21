from lib import *
import features, homography, hv, metrics, blend

def combine(img1, img2, method='SIFT', match='KNN', alpha=0.1):
    kpsA, featuresA = features.detectAndDescribe(img1, method=method)
    kpsB, featuresB = features.detectAndDescribe(img2, method=method)

    # features.showKeypoints(img1, img2, kpsA, kpsB) # Uncomment to keypoints
    print("Using: {} feature matcher".format(match))
    matches = features.showFeatureMatch(img1, img2, kpsA, kpsB, 
                                        featuresA, featuresB, method, match)
    matches = sorted(matches, key = lambda x:x.distance)
    nmatches = len(matches)
    if (nmatches > 100):
        threshold = matches[:int(nmatches*alpha)]
        matches = threshold
    
    mask = homography.getHomography(kpsA, kpsB, matches, 5.0)
    if mask is None:
        print("Not enough matches are found - {}/{}".format(len(matches), 4))
        return img1
    
    (M, mask) = mask
    
    # Warping image 1 to image 2
    warped_img1 = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    return warped_img1

if __name__ == '__main__':
    # Taking input parameters
    feature_extractor = input("Enter the feature extractor:\n1. SIFT\n2. AKAZE\n3. BRISK\n4. ORB\n")
    if (feature_extractor == '1'):
        feature_extractor = 'SIFT'
    elif (feature_extractor == '2'):
        feature_extractor = 'AKAZE'
    elif (feature_extractor == '3'):
        feature_extractor = 'BRISK'
    else:
        feature_extractor = 'ORB'

    feature_matching = input("Enter the feature matching method:\n1. Brute Force\n2. KNN\n3. FLANN\n")
    if (feature_matching == '1'):
        feature_matching = 'BF'
    elif (feature_matching == '2'):
        feature_matching = 'KNN'
    elif (feature_matching == '3'):
        feature_matching = 'FLANN'
    
    n = int(input("Number of samples: "))
    alpha = float(input("Alpha value: "))
    alpha = alpha/100
    
    # Asserting the input parameters
    assert feature_extractor in ['AKAZE', 'SIFT', 'BRISK', 'ORB'], "Invalid feature extractor"
    assert feature_matching in ['BF', 'KNN','FLANN'], "Invalid feature matching method"

    # Reading the images
    img1 = cv2.imread('../fisb_dataset/sub/scene_1/1.jpeg')
    # asserting image is not empty
    assert img1 is not None, "Image not found"
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img_map = img1
    print("Running stitching algorithm for {} images...".format(n))
    for i in tqdm(range(2,n+1)):
        path = '../fisb_dataset/sub/scene_1/{}.jpeg'
        img = cv2.imread(path.format(i))
        # asserting image is not empty
        assert img is not None, "Image not found"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oImg = combine(img_map, img, feature_extractor, feature_matching, alpha)

        # Blending the two imaegs
        result = blend.seamless_cloning(oImg, img)
        img_map = result

    print("Finshed stitching.\n")

    img_map = cv2.cvtColor(img_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite('../output/scene_1.png', img_map)

    orgImg = cv2.imread('../fisb_dataset/super/scene_1.jpeg')
    # Showing Metrics
    metrics.showStatistics(img_map, orgImg, feature_extractor, feature_matching, alpha)