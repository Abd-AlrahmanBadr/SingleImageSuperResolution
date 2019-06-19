# Graduation-Project

# Single Image Super Resolution Using Residual Blocks In GAN (Generative Adversarial Network)

**Single Image Super-Resolution (SISR)** is a notoriously challenging ill-posed problem, which aims to obtain a High-Resolution(HR) output from one of its Low-Resolution(LR) versions. This problem is quite complex since there exist multiple solutions for a given Low-Resolution image. To solve the SISR problem, a recently powerful deep learning algorithms have been employed and achieved the state-of-the-art performance. Most of current SISR solutions are based on image processing, the ways vary and their accuracies. Deep Learning provided a great solution with better accuracies than normal and basic image processing techniques. The most distinguished algorithm is Conditional Adversarial Nets of the Generative Adversarial Networks(GAN) model which has a great output. We propose a new improved technique based on the GAN architecture and Residual Blocks. Residual Block is an Artificial Neural Network (ANN) of a kind that builds on constructs known from pyramidal cells. Residual Neural Networks do this by utilizing skip connections. Skipping effectively simplifies the network, using fewer layers in the initial training stages. This speeds learning by reducing the impact of vanishing gradients, as there are fewer layers to propagate through. The network then gradually restores the skipped layers as it learns the feature space. We created **Residual-In-Residual(RIR)** blocks which consists of multiple residual blocks with the advantage of skip connections to preserve features from the previous blocks. Our loss function is based on the pre-trained VGG19 with the ImageNet weights.

## Full Project Details
This is project Full Documentation. [Link](https://github.com/Abd-AlrahmanBadr/Graduation-Project/blob/master/Documentation/Doc.pdf)

## GAN Model Structure
![alt text](/Documentation/Images/GANStructure.png  "GAN Model Structure")

## Residual Block Structure
![alt text](/Documentation/Images/ResidualStructure.png  "Residual Block Structure")

## Training
To prepare the training data or make out pre-processing for the model, the used time for cropping and saving the images for each resolution from the 900 images in [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Training I(HR) Resolution | 256\*256 | 128\*128 | 96\*96
--- | --- | --- | ---
Images Count | 31K | 136K | 260K
Cropping Time(minutes) | 2:26 | 3:43 | 5:31
Training Time(minutes/epoch) 2X | - | 64 | 80
Training Time(minutes/epoch) 4X | 7 | 50 | 75
Best PSNR-SSIM 2X | - | 26.73 - 0.76 | 28.15 - 0.58
Best PSNR-SSIM 4X | 21.52 - 0.47 | 21.99 - 0.52 | 22.52 - 0.0.58

**Note**: PSNR-SSIM Results were on 500 Images from Flickr2K Dataset. These results after finishing the training and deploying the model.

## Team Supervisor:
Assoc. Prof. Hala Abd-ElGalil (Head of Computer Science Department - Faculty of Computers and Information - Helwan University)

## Team Members:
- Abd-Alrahman Yousry Badr abdalr7man.yousry@gmail.com
- Ahmed El-Shafey Shaltout ahmedshaltout.fcih@gmail.com
- Mohamed Ashraf El-Melegy mohamedelmelegy0@outlook.com
- Ahmed Ayman El-Sayed ahmedayman190774@gmail.com
- Ahmed Sayed Hamed ahmedsaied.fcih@gmail.com
- Alaa Mohamed Ali alaaabozaid21@gmail.com
