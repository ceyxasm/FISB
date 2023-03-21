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
    # Warping image 2 to image 1
   #  warped_img2 = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1], img1.shape[0]))

    return warped_img1

if __name__ == '__main__':
    # Taking input parameters from argument
    feature_extractor = int(sys.argv[1])
    feature_matching = int(sys.argv[2])
    n = int(sys.argv[3]) # Number of samples
    alpha = int(sys.argv[4])
    dir = sys.argv[5]

    # feature_extractor = input("Enter the feature extractor:\n1. SIFT\n2. AKAZE\n3. BRISK\n4. ORB\n")
    if (feature_extractor == 1):
        feature_extractor = 'SIFT'
    elif (feature_extractor == 2):
        feature_extractor = 'AKAZE'
    elif (feature_extractor == 3):
        feature_extractor = 'BRISK'
    elif (feature_extractor == 4):
        feature_extractor = 'ORB'

    # feature_matching = input("Enter the feature matching method:\n1. Brute Force\n2. KNN\n3. FLANN\n")
    if (feature_matching == 1):
        feature_matching = 'BF'
    elif (feature_matching == 2):
        feature_matching = 'KNN'
    elif (feature_matching == 3):
        feature_matching = 'FLANN'
    
    # Asserting the input parameters
    assert feature_extractor in ['AKAZE', 'SIFT', 'BRISK', 'ORB'], "Invalid feature extractor"
    assert feature_matching in ['BF', 'KNN','FLANN'], "Invalid feature matching method"

    # Storing all files in dir in a list
    files = os.listdir(dir)
    files.sort()

    # Reading the images
    img1 = cv2.imread(os.path.join(dir, files[0]))
    
    # asserting image is not empty
    assert img1 is not None, "Image not found"
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img_map = img1
    print("Running stitching algorithm for {} images...".format(n))
    for i in tqdm(range(2,n+1)):
        img = cv2.imread(os.path.join(dir, files[i-1]))
        # asserting image is not empty
        assert img is not None, "Image not found"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oImg = combine(img_map, img, feature_extractor, feature_matching, alpha)

        # Blending the two imaegs
        result = blend.seamless_cloning(oImg, img)
        img_map = result


    print("Finshed stitching.\n")

    # Folder name
    folder = dir.split('/')[-2]

    # Saving the images in the output folder
    img_map = cv2.cvtColor(img_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join("output/", folder + ".png"), img_map)
    plt.close()

    # Orignal Image
    orgImg = cv2.imread(os.path.join("fisb_dataset/super/" + folder + ".jpeg"))
    assert orgImg is not None, "Image not found"

    # Showing Metrics
    metrics.showStatistics(img_map, orgImg, feature_extractor, feature_matching, alpha)