from utils.utils import CreateFolder, LoadImagesPaths
import cv2, argparse, os

def CropImage(Image, Size):
    """ Cropping Image """
    i = 0
    Images = []
    while(i + Size < Image.shape[0]):
        j = 0
        while(j + Size < Image.shape[1]):
            Images.append(Image[i:i + Size, j: j + Size, ])
            j += Size
        i += Size
    
    return Images

def CropImages(ImagesPaths, Size):
    SaveDirName = "./TrainDataset/"
    CreateFolder(SaveDirName)

    ImagesCountLength = len(str(len(ImagesPaths) * 30))
    i = 1
    for ImagePath in ImagesPaths:
        Image = cv2.imread(ImagePath)
        Images = CropImage(Image, Size)
        for Image in Images:
            cv2.imwrite(SaveDirName + '0' * (ImagesCountLength - len(str(i))) + str(i) + ".png", Image)
            i += 1

def ResizeImages(ImagesPaths, Size):
    SaveDirName = "./DatasetResized_{}/".format(Size)
    CreateFolder(SaveDirName)
    
    ImagesCountLength = len(str(len(ImagesPaths)))
    i = 1
    for ImagePath in ImagesPaths:
        Image = cv2.imread(ImagePath)
        Max_HW = min(Image.shape[0], Image.shape[1])
        Image = Image[:Max_HW, :Max_HW, :]
        Image = cv2.resize(Image, (Size, Size))
        cv2.imwrite(SaveDirName + '0' * (ImagesCountLength - len(str(i))) + str(i) + ".png", Image)
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-directories', type = str, default = os.getcwd() + '/Dataset/', help = 'Directories Paths of Dataset')
    parser.add_argument('-s', '--size-image', type = int, default = 256, help = 'PreProcess Image to Size')
    parser.add_argument('-t', '--preprocessing-type', type = int, default = 1, help = 'PreProcessing Type; 1 >> Cropping, 2 >> Resizing')

    FLAGS, unparsed = parser.parse_known_args()

    ImagesPaths = []
    for Directory in FLAGS.dataset_directories.split(','):
        ImagesPaths += LoadImagesPaths(Directory)

    if FLAGS.preprocessing_type == 1:
        CropImages(ImagesPaths, FLAGS.size_image)
    elif FLAGS.preprocessing_type == 2:
        ResizeImages(ImagesPaths, FLAGS.size_image)

# python PreProcessDataset.py -d ./Dataset/ -s 128 -t 1