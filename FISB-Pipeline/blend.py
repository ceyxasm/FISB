from lib import *

def alpha_blend(img1, img2):
    # Create a linearly increasing mask
    mask = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)
    mask[:, :int(img2.shape[1]/2)] = np.linspace(0, 1, int(img2.shape[1]/2))
    mask = mask.astype(np.float32)
    mask = 0.5
    result = cv2.addWeighted(img1, float(1-mask), img2, float(mask), 0)
    return result

def gaussian_blend(img1, img2):
    # Compute the Gaussian pyramid for each image
    G1 = img1.copy()
    G2 = img2.copy()
    gp1 = [G1]
    gp2 = [G2]
    for i in range(6):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        gp1.append(G1)
        gp2.append(G2)

    # Compute the Laplacian pyramid for each image
    lp1 = [gp1[5]]
    lp2 = [gp2[5]]
    for i in range(5, 0, -1):
        size = (gp1[i - 1].shape[1], gp1[i - 1].shape[0])
        L1 = cv2.subtract(gp1[i - 1], cv2.pyrUp(gp1[i], dstsize=size))
        L2 = cv2.subtract(gp2[i - 1], cv2.pyrUp(gp2[i], dstsize=size))
        lp1.append(L1)
        lp2.append(L2)

    # Combine the left and right halves of each level of the Laplacian pyramid
    LS = []
    for l1, l2 in zip(lp1, lp2):
        rows, cols, dpt = l1.shape
        ls = np.hstack((l1[:, :cols//2], l2[:, cols//2:]))
        LS.append(ls)

    # Reconstruct the blended image from the Laplacian pyramid
    ls_ = LS[0]
    for i in range(1, 6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.add(cv2.pyrUp(ls_, dstsize=size), LS[i])

    result = ls_

    return result

def seamless_cloning(img1, img2):
    # Create a mask for the center of the image
    mask = np.zeros(img1.shape, dtype=np.uint8)
    mask[img1.shape[0]//2-50:img1.shape[0]//2+50, img1.shape[1]//2-50:img1.shape[1]//2+50] = 255

    # Create a rough mask around the center of the image
    rough_mask = np.zeros(img1.shape, dtype=np.uint8)
    rough_mask[img1.shape[0]//2-100:img1.shape[0]//2+100, img1.shape[1]//2-100:img1.shape[1]//2+100] = 255

    # Use the rough mask to find the center of the image
    M = cv2.moments(rough_mask[:,:,0])
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # Use the center to create a seamless cloning mask
    result = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)

    return result

# Multi-band blending
"""
OPENCV PYTHON:
void cv::detail::MultiBandBlender::feed	(	InputArray 	img,
InputArray 	mask,
Point 	tl 
)	"""
def multiband_blend(img1, img2):
    # Create a rough mask around the center of the image
    rough_mask = np.zeros(img1.shape, dtype=np.uint8)
    rough_mask[img1.shape[0]//2-100:img1.shape[0]//2+100, img1.shape[1]//2-100:img1.shape[1]//2+100] = 255

    # Convert the rough_mask to the required type (8-bit unsigned integer)
    rough_mask = rough_mask.astype(np.uint8)

    # Use the rough mask to find the center of the image
    M = cv2.moments(rough_mask[:,:,0])
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # Create a multiband blender
    blender = cv2.detail_MultiBandBlender()
    print(rough_mask.shape, center)
    # Blend the images with the Assertion failed) mask.type() == CV_8U in function 'feed'
    blender.feed(img1, rough_mask, center)
    blender.feed(img2, rough_mask, center)
    result = blender.blend(rough_mask, center)

    return result
