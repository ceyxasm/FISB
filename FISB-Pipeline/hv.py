from lib import *

def hStich(img1, img2, H):
    # Apply a horizontal panorama
    width = img2.shape[1] + img1.shape[1]
    height = max(img2.shape[0], img1.shape[0])
    # otherwise, apply a perspective warp to stitch the images
    # together
    result = cv2.warpPerspective(img1, H, (width, height))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    plt.figure(figsize=(20,10))
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.savefig('../output/himg.png')
    plt.close()
    return result

def vStitch(img1, img2, H):
    # Apply a vertical panorama
    width = max(img1.shape[1], img2.shape[1])
    height = img1.shape[0] + img2.shape[0]

    result = cv2.warpPerspective(img1, H, (width, height))
    result[img1.shape[0]:height, 0:img1.shape[1]] = img2

    plt.figure(figsize=(20,10))
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.savefig('../output/vimg.png')
    plt.close()
    return result