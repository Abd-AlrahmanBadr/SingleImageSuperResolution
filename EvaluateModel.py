import math, os, cv2, argparse, time
import numpy as np
from skimage.measure import compare_ssim
from model.generator import LoadGeneratorWithWeights
import tensorflow as tf

def LoadImagesPaths(Path):
    images = []
    for imageName in os.listdir(Path):
        images.append(Path + imageName)
    return images[:10]

def MSE(HR, SR):
	return np.mean((HR - SR) ** 2)

def PSNR(_MSE):
	if _MSE == 0:
		return 100

	return 20 * math.log10(255.0 / math.sqrt(_MSE))

def SSIM(HR, SR):
	return compare_ssim(HR, SR, multichannel = True)

def EvaluateModel(model, graph, ImagesPaths, ScalingFactor):
    TotalP = 0.0
    TotalS = 0.0
    TotalM = 0.0
    TotalT = 0.0

    print("Testing {} Images".format(len(ImagesPaths)))

    i = 1
    with graph.as_default():
        for ImagePath in ImagesPaths:
            start = time.time()

            HR = Image = cv2.imread(ImagePath)
            LR = cv2.resize(Image, (Image.shape[1] // ScalingFactor, Image.shape[0] // ScalingFactor), interpolation = cv2.INTER_CUBIC)

            SR = model.predict(np.expand_dims(LR, axis = 0))
            SR = SR[0]

            # cv2.imshow("SR", SR)
            # cv2.waitKey(0)

            M = MSE(HR, SR)
            P = PSNR(M)
            S = SSIM(HR, SR)

            TotalP += P
            TotalS += S
            TotalM += M
            TotalT += time.time() - start

            print("Finished Image #{}".format(i))
            i += 1
    
    AverageP = TotalP / len(ImagesPaths)
    AverageM = TotalM / len(ImagesPaths)
    AverageS = TotalS / len(ImagesPaths)
    AverageT = TotalT / len(ImagesPaths)

    print("Average PSNR: ", AverageP)
    print("Average SSIM: ", AverageS)
    print("Average MSE: ", AverageM)
    print("Average Time: ", AverageT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-directories', type = str, default = os.getcwd() + '/TestDataset/', help = 'Directories Paths of Dataset')
    parser.add_argument('-m', '--model-path', type = str, default = os.getcwd() + '/TrainedModel/Epoch_1/Generator.h5', help = 'Saved Generator Path')
    parser.add_argument('-f', '--scaling-factor', type = int, default = 2, help = 'Scaling Factor for Test')

    FLAGS, unparsed = parser.parse_known_args()

    ImagesPaths = []
    for Directory in FLAGS.dataset_directories.split(','):
        ImagesPaths += LoadImagesPaths(Directory)

    model = LoadGeneratorWithWeights(FLAGS.model_path, FLAGS.scaling_factor)
    graph = tf.get_default_graph()

    EvaluateModel(model, graph, ImagesPaths, FLAGS.scaling_factor)

# python EvaluateModel.py -d ./TestDataset/ -f 2 -m ./TrainedModel/Epoch_10/Generator.h5