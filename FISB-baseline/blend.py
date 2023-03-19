from lib import *

def alpha_blend(img1, img2):
    # Create a linearly increasing mask
    mask = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)
    mask[:, :int(img2.shape[1]/2)] = np.linspace(0, 1, int(img2.shape[1]/2))

    # Convert the mask to a floating-point data type
    mask = mask.astype(np.float32)
    mask = 0.5
    # Blend the two images using alpha blending
    result = cv2.addWeighted(img1, float(1-mask), img2, float(mask), 0)

    return result

def gaussian_blend(img1, img2):
    # # Create a linearly increasing mask
    # mask = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)
    # mask[:, :int(img2.shape[1]/2)] = np.linspace(0, 1, int(img2.shape[1]/2))

    # # Convert the mask to a floating-point data type
    # mask = mask.astype(np.float32)

    # # Create a Gaussian mask
    # mask = cv2.GaussianBlur(mask, (0, 0), 5)

    # # Blend the two images using alpha blending
    # result = cv2.addWeighted(img1, float(1-mask), img2, float(mask), 0)

    # Create a binary mask of the same size as the input images
    mask = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)
    mask[:, :int(img2.shape[1]/2)] = 255

    # Build the Laplacian pyramids of the input images and the mask
    img1_pyr = cv2.buildLaplacianPyramid(img1, 6)
    img2_pyr = cv2.buildLaplacianPyramid(img2, 6)
    mask_pyr = cv2.buildLaplacianPyramid(mask, 6)

    # Blend the Laplacian pyramids of the input images using the mask pyramid
    blended_pyr = []
    for i in range(len(img1_pyr)):
        blended_pyr.append(img1_pyr[i] * mask_pyr[i] + img2_pyr[i] * (1 - mask_pyr[i]))

    # Reconstruct the blended image from the blended Laplacian pyramid
    blended_img = cv2.pyrUp(blended_pyr[-1])
    for i in range(len(blended_pyr)-2, -1, -1):
        blended_img = cv2.pyrUp(blended_img)
        blended_img = cv2.add

    result = blended_img

    return result
