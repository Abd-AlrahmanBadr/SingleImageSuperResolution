import argparse, os
from model.gan import SRGAN
from utils.utils import LoadImages

def TrainModel(DatasetPath, ScaleFactor, nBaseBlocks, nResidualBlocks, VisualizeModel, BatchSize, EpochsCount, SavingDirPath, ModelSavingInterval):
    HR_Images, LR_Images = LoadImages(DatasetPath, ScaleFactor)

    print("------------------------------------")
    print("HR Data Shape: ", HR_Images.shape)
    print("LR Data Shape: ", LR_Images.shape)
    print("------------------------------------")

    Model = SRGAN(HR_Images.shape[1:], LR_Images.shape[1:], nBaseBlocks, nResidualBlocks)

    if(VisualizeModel):
        Model.ViewModelSummary()

    Model.Train(HR_Images, LR_Images, BatchSize, EpochsCount, SavingDirPath, ModelSavingInterval)

if __name__== "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-path', type = str, default = os.getcwd() + './TrainDataset/', help = 'Dataset Path')
    parser.add_argument('-f', '--scale-factor', type = int, default = 4, help = 'UpScaling Factor')
    parser.add_argument('-b', '--n-base-blocks', type = int, default = 6, help = 'Number of Base Blocks')
    parser.add_argument('-r', '--n-residual-blocks', type = int, default = 2, help = 'Number of Base Blocks')
    parser.add_argument('-v', '--visualize-model', type = int, default = 1, help = 'Visualize Model Structure & Save figs to VisualizeModel Folder')
    parser.add_argument('-s', '--batch-size', type = int, default = 16, help = 'Batch Size for Training')
    parser.add_argument('-e', '--epochs-count', type = int, default = 100, help = 'Epochs Count for Training')
    parser.add_argument('-m', '--model-saving-dir', type = str, default = "./TrainedModel/", help = 'Saving Trained Model to Directory')
    parser.add_argument('-i', '--saving-interval', type = int, default = 1, help = 'Saving Model Every nEpochs')

    FLAGS, unparsed = parser.parse_known_args()

    TrainModel(
        FLAGS.dataset_path, 
        FLAGS.scale_factor, 
        FLAGS.n_base_blocks,
        FLAGS.n_residual_blocks, 
        FLAGS.visualize_model, 
        FLAGS.batch_size, 
        FLAGS.epochs_count, 
        FLAGS.model_saving_dir, 
        FLAGS.saving_interval
    )

# python train.py -d ./TrainDataset/ -b 3 -r 6 -f 2 -m ./TrainedModel/