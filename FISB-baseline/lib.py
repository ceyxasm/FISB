import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.metrics import structural_similarity as Compare_SSIM
from tqdm import tqdm
cv2.ocl.setUseOpenCL(False)