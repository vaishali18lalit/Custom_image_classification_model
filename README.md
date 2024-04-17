# Custom_image_classification_model
custom-built image classification model designed to identify various road surfaces, a key innovation for autonomous vehicle technology and infrastructure management.
# Autoencoder for Image Classification
# Overview
This repository contains the implementation of an innovative autoencoder model tailored for image classification tasks. The model combines the proven feature extraction capabilities of AlexNet with the versatility of an encoder-decoder architecture, resulting in enhanced accuracy and interpretability in distinguishing various road surfaces.

# Model Architecture
The model architecture comprises two main components: the Encoder and the Decoder.

# Encoder
Input Layer: Processes images of size 224x224 pixels with 3 channels (RGB).
First Convolutional Layer: Utilizes 96 filters with an 11x11 kernel size and a stride of 4.
First Max Pooling Layer: Performs down-sampling with a 3x3 pool size and 2-stride.
Second Convolutional Layer: Employs 256 filters of a 5x5 kernel size with 'same' padding.
Second Max Pooling Layer: Further reduces feature map size with a 3x3 pool size and 2-stride.
Additional Convolutional Layers: Three sets of convolutional layers with increasing filter depth (384, 384, and 256).
Final Encoder Convolutional Layer: Utilizes 256 filters with a 3x3 kernel size and 'same' padding.
Decoder
Conv2D Transpose Layers: Reverse convolutional operations with varying filter sizes to perform up-sampling.
Upsampling Layers: Used to further increase the size of feature maps.
Final Conv2D Transpose Layer: Applies a final up-sampling to nearly restore the original image size.
Flatten Layer: Converts multi-dimensional tensor to a single vector.
Dense Layers: Two dense layers with 4096 units each and ReLU activation.
Output Layer: Final dense layer with 3 units and softmax activation for class probability distribution.
# Training
The model is trained using the entire dataset. Training involves minimizing the reconstruction error between the original image and its reconstruction, which fine-tunes the model parameters to preserve essential information during encoding.

# Inference
After training, the encoder is used for inference in the classification task. This encoder-decoder architecture facilitates capturing complex features from images, enabling the encoder alone to perform better on the classification task.

# Tensorboard Graphs
Tensorboard graphs are provided to visualize the training and validation loss and accuracy over epochs.
