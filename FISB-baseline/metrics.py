from lib import *
import features, homography

def getAlignedImage(img1, img2, method, match, alpha):
    # Align both images using homography
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

    img1_aligned = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    return img1_aligned

# Metrics to compare the images
def showStatistics(img1, img2, method, match, alpha):
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Align both images using homography
    img1_g = getAlignedImage(img1_g, img2_g, method, match, alpha)

    print("Statistics of stitched image with orignal image:")
    # MSE between images
    mse = np.mean((img1_g - img2_g) ** 2)
    print("mse:",mse)

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