from lib import *
import features, homography, hv, metrics, blend

def combine(img1, img2, method, match, alpha):
    kpsA, featuresA = features.detectAndDescribe(img1, method=method)
    kpsB, featuresB = features.detectAndDescribe(img2, method=method)

    features.showKeypoints(img1, img2, kpsA, kpsB)
    print("Using: {} feature matcher".format(match))
    matches = features.showFeatureMatch(img1, img2, kpsA, kpsB, featuresA, featuresB, method, match)

    # Identifying good matches
    good_matches_count = int(len(matches) * alpha)
    print("Good matches: {}".format(good_matches_count))
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * alpha)]

    # Homography of the images
    M = homography.getHomography(kpsA, kpsB, good_matches, thresh=5)
    
    if M is None:
        print("Error!")
    (good_matches, H, status) = M

    # Warping image 1 to image 2
    warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    # Blending the images
    result = blend.alpha_blend(warped_img1, img2)

    return result

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

    feature_matching = input("Enter the feature matching method:\n1. Brute Force\n2. Flann\n")
    if (feature_matching == '1'):
        feature_matching = 'BF'
    else:
        feature_matching = 'Flann'
    
    n = int(input("Number of samples: "))
    alpha = float(input("Alpha value: "))
    
    # Asserting the input parameters
    assert feature_extractor in ['AKAZE', 'SIFT', 'BRISK', 'ORB'], "Invalid feature extractor"
    assert feature_matching in ['BF', 'Flann'], "Invalid feature matching method"

    # Reading the images
    img1 = cv2.imread('../test/1.jpeg')
    # asserting image is not empty
    assert img1 is not None, "Image not found"
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    img_map = img1
    img_map_gray = img1_gray
    print("Running stitching algorithm for {} images...".format(n))
    for i in tqdm(range(2,n+1)):
        path = '../test/{}.jpeg'
        img = cv2.imread(path.format(i))
        # asserting image is not empty
        assert img is not None, "Image not found"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_map = combine(img_map, img, feature_extractor, feature_matching, alpha)

    print("Finshed stitching.\n")

    # Plotting the final image_map
    plt.figure(figsize=(20,10))
    plt.imshow(img_map)
    plt.axis('off')
    plt.savefig('../output/img_map.png')
    plt.close()

    # Showing Metrics
    metrics.showStatistics(img_map, img1, feature_extractor, feature_matching, alpha)