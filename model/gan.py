from model.generator import Generator
from model.discriminator import Discriminator

from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from utils.utils import CreateFolder, PSNR, SSIM, MSE
from utils.utils_model import ContentLoss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2, time

class SRGAN:
    def __init__(self, HR_Shape, LR_Shape, nBaseBlocks = 6, nResidualBlocks = 2):
        self.HR_Shape = HR_Shape
        self.LR_Shape = LR_Shape
        ScaleFactor = self.HR_Shape[0] // self.LR_Shape[0]

        self.Generator = Generator(nBaseBlocks, nResidualBlocks, ScaleFactor).BuildGenerator()
        self.Discriminator = Discriminator(HR_Shape).BuildDiscriminator()
        
        AdamOptimizer = Adam(lr = 1E-4, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
        
        self.Discriminator.compile(loss = "binary_crossentropy", optimizer = AdamOptimizer, metrics = [])
        self.Discriminator.trainable = False

        GAN_input = Input(shape = self.LR_Shape)
        x = self.Generator(GAN_input)
        GAN_output = self.Discriminator(x)
        self.GAN = Model(inputs = GAN_input, outputs = [x, GAN_output])
        self.GAN.compile(
            loss = [ContentLoss, "binary_crossentropy"],
            loss_weights = [0.006, 0.001], 
            optimizer = AdamOptimizer
        )

    def ViewModelSummary(self):
        CreateFolder("ModelVisualization")

        # Generator Model
        print('-' * 20, "Generator Model", '-' * 20)
        print(self.Generator.summary())
        plot_model(self.Generator, to_file = "./ModelVisualization/Generator.png")

        # Discriminator Model
        print('-' * 20, "Discriminator Model", '-' * 20)
        print(self.Discriminator.summary())
        plot_model(self.Discriminator, to_file = "./ModelVisualization/Discriminator.png")

        # GAN Model
        print('-' * 20, "GAN Model", '-' * 20)
        print(self.GAN.summary())
        plot_model(self.GAN, to_file = "./ModelVisualization/GAN.png")
    
    def PredictSample(self, SavingDir, n_imgs = 3):
        plt.figure(figsize = (50, 50))
        plt.tight_layout()
        for i in range(0, n_imgs * 3, 3):
            idx = np.random.randint(0, self.Test_LR.shape[0] - 1)
            LR_Image = self.Test_LR[idx]
            HR_Image = self.Test_HR[idx]

            plt.subplot(n_imgs, 3, i + 1)
            plt.imshow(HR_Image)
            plt.axis('off')
            plt.title('HR Image')

            plt.subplot(n_imgs, 3, i + 2)
            plt.imshow(cv2.resize(LR_Image, self.HR_Shape[:2], interpolation = cv2.INTER_CUBIC))
            plt.axis('off')
            plt.title('X4 (Bicubic) Image')

            SR_Image = self.Generator.predict(np.expand_dims(LR_Image, axis = 0))
            plt.subplot(n_imgs, 3, i + 3)
            plt.imshow(np.squeeze(SR_Image, axis = 0).astype(np.uint8))
            plt.axis('off')
            plt.title('SR Image')

        plt.savefig(SavingDir + 'Sample.png')
        plt.clf()
    
    def Train(self, HR_Images, LR_Images, BatchSize, EpochsCount, SavingDirPath, ModelSavingInterval):
        split_size = int(len(HR_Images) * 0.9)
        Train_HR, self.Test_HR = HR_Images[:split_size], HR_Images[split_size:]
        Train_LR, self.Test_LR = LR_Images[:split_size], LR_Images[split_size:]

        self.n_TestImages = self.Test_HR.shape[0]

        print("------------------------------------")
        print("Train HR Data Shape: ", Train_HR.shape)
        print("Train LR Data Shape: ", Train_LR.shape)
        print("Test HR Data Shape: ", self.Test_HR.shape)
        print("Test LR Data Shape: ", self.Test_LR.shape)
        print("------------------------------------")

        CreateFolder(SavingDirPath)

        loss_file = open(SavingDirPath + 'Losses.txt' , 'w+')
        loss_file.close()

        n_batch = int(Train_HR.shape[0] / BatchSize)

        for e in range(1, EpochsCount + 1):
            StartTime = time.time()

            print('-' * 20, "Epoch #{}".format(e), '-' * 20)
            for _ in tqdm(range(n_batch)):
                imgs_indexes = np.random.randint(0, Train_HR.shape[0], size = BatchSize)
                batch_HRImages = Train_HR[imgs_indexes]
                batch_LRImages = Train_LR[imgs_indexes]

                GeneratedImages_SR = self.Generator.predict(batch_LRImages)

                real_data_Y = np.ones(BatchSize) + 0.05 * np.random.random(BatchSize)
                fake_data_Y = np.zeros(BatchSize) + 0.05 * np.random.random(BatchSize)

                self.Discriminator.trainable = True

                d_loss_real = self.Discriminator.train_on_batch(batch_HRImages, real_data_Y)
                d_loss_fake = self.Discriminator.train_on_batch(GeneratedImages_SR, fake_data_Y)
                Discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                
                imgs_indexes = np.random.randint(0, Train_HR.shape[0], size = BatchSize)
                batch_HRImages = Train_HR[imgs_indexes]
                batch_LRImages = Train_LR[imgs_indexes]

                GAN_y = np.ones(BatchSize)
                self.Discriminator.trainable = False
                GAN_loss = self.GAN.train_on_batch(batch_LRImages, [batch_HRImages, GAN_y])
            
            print("Discriminator Loss in Epoch #{} = {}".format(e, Discriminator_loss))
            print("GAN Loss in Epoch #{} = {}".format(e, GAN_loss))
            AveragePSNR, AverageSSIM, AverageMSE = self.EvaluateModelEpoch()
            print("GAN PSNR Epoch #{} = {}".format(e, AveragePSNR))
            print("GAN SSIM Epoch #{} = {}".format(e, AverageSSIM))
            print("GAN MSE Epoch #{} = {}".format(e, AverageMSE))

            if e % ModelSavingInterval == 0:
                EpochFolder = SavingDirPath + "Epoch_{}/".format(e)
                CreateFolder(EpochFolder)

                self.PredictSample(EpochFolder)
                self.Generator.save(EpochFolder + "Generator.h5")
                self.Discriminator.save(EpochFolder + "Discriminator.h5")
                self.GAN.save(EpochFolder + "GAN.h5")
            
            UsedTime = time.time() - StartTime
            
            loss_file = open(SavingDirPath + 'Losses.txt' , 'a')
            loss_file.write("Epoch #{}: GAN_loss = {}; Discriminator_loss = {}; Average PSNR = {}; Average SSIM = {}; Avaerage MSE = {}; Total Time = {}\n".format(e, GAN_loss, Discriminator_loss, AveragePSNR, AverageSSIM, AverageMSE, UsedTime))
            loss_file.close()

    def EvaluateModelEpoch(self):
        TotalPSNR = 0.0
        TotalSSIM = 0.0
        TotalMSE = 0.0

        for i in range(self.n_TestImages):
            LR_Image = self.Test_LR[i]
            HR_Image = self.Test_HR[i]

            GeneratedImage_SR = self.Generator.predict(np.expand_dims(LR_Image, axis = 0))

            TotalPSNR += PSNR(HR_Image, GeneratedImage_SR[0])
            TotalSSIM += SSIM(HR_Image, GeneratedImage_SR[0])
            TotalMSE += MSE(HR_Image, GeneratedImage_SR[0])
        
        return TotalPSNR / self.n_TestImages, TotalSSIM / self.n_TestImages, TotalMSE / self.n_TestImages