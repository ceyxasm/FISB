from lib import *
import features, homography

def align(img1, img2, method, match, alpha):
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
        return np.resize(img1, img2.shape)
    
    (M, mask) = mask
    
    # Warping image 1 to image 2
    warped_img1 = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    return warped_img1

# Metrics to compare the images
def showStatistics(img1, img2, method='SIFT', match='BF', alpha=0.1):
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Align both images using homography
    img1_g = align(img1_g, img2_g, method, match, alpha)

    print("\nStatistics of stitched image with orignal image:")
    # RMSE between images
    rmse = np.sqrt(np.mean((img1_g - img2_g) ** 2))
    print("mse:",rmse)

    # SSIM between images
    ssim = Compare_SSIM(img1_g, img2_g)
    print("ssim:",ssim)

    # PSNR between images
    psnr = cv2.PSNR(img1_g, img2_g)
    print("psnr:",psnr)

    # NCC between images
    ncc = cv2.matchTemplate(img1_g, img2_g, cv2.TM_CCORR_NORMED)
    print("ncc:",ncc[0][0])
    
    # NMI between images
    nmi = cv2.matchTemplate(img1_g, img2_g, cv2.TM_CCOEFF_NORMED)
    print("nmi:",nmi[0][0])