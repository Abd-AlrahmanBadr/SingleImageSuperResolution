import math, os, cv2
import numpy as np 
from skimage.measure import compare_ssim
import tensorflow as tf
import random

def LoadImagesPaths(Path):
	images = []
	for imageName in os.listdir(Path):
		images.append(Path + imageName)
	
	random.shuffle(images)
	return images

def LoadImages(Path, ScaleFactor):
	HR_Images = []
	LR_Images = []

	for ImagePath in LoadImagesPaths(Path):
		Image = cv2.imread(ImagePath)
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
		HR_Images.append(Image)
		LR_Size = Image.shape[0] // ScaleFactor
		LR_Images.append(cv2.resize(Image, (LR_Size, LR_Size)))
	
	return np.array(HR_Images), np.array(LR_Images)

def CreateFolder(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def MSE(HR, SR):
	return np.mean((HR - SR) ** 2)

def PSNR(HR, SR):
	_MSE = MSE(HR, SR)
	if _MSE == 0:
		return 100

	return 20 * math.log10(255.0 / math.sqrt(_MSE))

def SSIM(originalImage, newImage):
	return compare_ssim(originalImage, newImage, multichannel = True)