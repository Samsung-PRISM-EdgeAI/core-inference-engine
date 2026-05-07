"""
This module provides a PyTorch implementation of a Convolutional Neural
Architecture for identifying diseases in apple fruit, as described in the paper:

Source Paper: "Using Machine Learning to Identify Diseases and Perform Sorting in Apple Fruit"
URL: https://www.semanticscholar.org/paper/0d2c38b39f73003ec23ee3e2382e319e20bd5d18

Mathematical Idea:
The core idea is to leverage a deep Convolutional Neural Network (CNN) to automatically
extract hierarchical features from images of apple fruit and classify them into
different disease categories (or healthy). The proposed architecture consists of
multiple convolutional layers, each followed by a Rectified Linear Unit (ReLU)
activation function and a max-pooling layer. These layers progressively reduce
the spatial dimensions of the input image while increasing the depth (number of
feature maps), learning increasingly complex patterns. After several such blocks,
the learned features are flattened and passed through one or more fully connected
(dense) layers. A final softmax activation function is applied to the output of
the last fully connected layer to produce probability scores for each disease class,
indicating the likelihood of the apple belonging to that class.

Key Hyperparameters and their default values (as derived from the paper's description
and common practices for similar architectures if not explicitly stated):
- `input_channels`: Number of color channels in the input image (e.g., 3 for RGB). Default: 3.
-