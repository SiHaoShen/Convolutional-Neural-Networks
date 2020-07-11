# Convolutional Neural Networks Project Notes

   * [Convolutional Neural Networks](#convolutional-neural-networks)
      * [Convolutional Neural Networks: Step by Step](#convolutional-neural-networks-step-by-step)
      * [Convolutional Neural Networks: Application](#convolutional-neural-networks-application)
      * [Keras tutorial - the Happy House](#keras-tutorial---the-happy-house)
      * [Residual Networks](#residual-networks)
      * [Art generation with Neural Style Transfer](#deep-learning--art-neural-style-transfer)
      * [Autonomous driving - Car detection](#autonomous-driving---car-detection)
	  
This repository is the summaries of the project Convolutional Neural Networks on on [DeepLearning.ai](https://deeplearning.ai) specialization courses.

## Convolutional Neural Networks

### Convolutional Neural Networks: Step by Step

Welcome to Course 4's first assignment! In this assignment, you will implement convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and (optionally) backward propagation. 

**Notation**:
- Superscript <sup>[l]</sup> denotes an object of the `l.th` layer.
    - Example: a<sup>[4]</sup> is the `4.th` layer activation. W<sup>[5]</sup> and b<sup>[5]</sup> are the `5.<sup>th</sup>` layer parameters.


- Superscript <sup>(i)</sup> denotes an object from the `i.th` example.
    - Example: x<sup>(i)</sup> is the `i.th` training example input.

- Lowerscript <sub>i</sub> denotes the `i.th` entry of a vector.
    - Example: a<sup>[l]</sup><sub>i</sub> denotes the `i.th` entry of the activations in layer `l`, assuming this is a fully connected (FC) layer.

- `n_H`, `n_W` and `n_C` denote respectively the height, width and number of channels of a given layer. If you want to reference a specific layer `l`, you can also write n_H<sup>[l]</sup>, n_W<sup>[l]</sup>, n_C<sup>[l]</sup>.
- `n_H_prev`, `n_W_prev` and `n_C_prev` denote respectively the height, width and number of channels of the previous layer. If referencing a specific layer `l`, this could also be denoted n_H<sup>[l-1]</sup>, n_W<sup>[l-1]</sup>, n_C<sup>[l-1]</sup>.

We assume that you are already familiar with `numpy` and/or have completed the previous courses of the specialization. Let's get started!

### Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Implement helper functions that you will use when implementing a TensorFlow model
- Implement a fully functioning ConvNet using TensorFlow 

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a classification problem 

We assume here that you are already familiar with TensorFlow. If you are not, please refer the *TensorFlow Tutorial* of the third week of Course 2 ("*Improving deep neural networks*").

### Keras tutorial - the Happy House

Welcome to the first assignment of week 2. In this assignment, you will:
1. Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK. 
2. See how you can in a couple of hours build a deep learning algorithm.

Why are we using Keras? Keras was developed to enable deep learning engineers to build and experiment with different models very quickly. Just as TensorFlow is a higher-level framework than Python, Keras is an even higher-level framework and provides additional abstractions. Being able to go from idea to result with the least possible delay is key to finding good models. However, Keras is more restrictive than the lower-level frameworks, so there are some very complex models that you can implement in TensorFlow but not (without more difficulty) in Keras. That being said, Keras will work fine for many common models. 

In this exercise, you'll work on the "Happy House" problem, which we'll explain below. Let's load the required packages and solve the problem of the Happy House!

### Residual Networks

Welcome to the second assignment of this week! You will learn how to build very deep convolutional networks, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allow you to train much deeper networks than were previously practically feasible.

**In this assignment, you will:**
- Implement the basic building blocks of ResNets. 
- Put together these building blocks to implement and train a state-of-the-art neural network for image classification. 

This assignment will be done in Keras. 

Before jumping into the problem, let's run the cell below to load the required packages.

### Deep Learning & Art: Neural Style Transfer

Welcome to the second assignment of this week. In this assignment, you will learn about Neural Style Transfer. This algorithm was created by Gatys et al. (2015) (https://arxiv.org/abs/1508.06576). 

**In this assignment, you will:**
- Implement the neural style transfer algorithm 
- Generate novel artistic images using your algorithm 

Most of the algorithms you've studied optimize a cost function to get a set of parameter values. In Neural Style Transfer, you'll optimize a cost function to get pixel values!

### Autonomous driving - Car detection

Welcome to your week 3 programming assignment. You will learn about object detection using the very powerful YOLO model. Many of the ideas in this notebook are described in the two YOLO papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242). 

**You will learn to**:
- Use object detection on a car detection dataset
- Deal with bounding boxes

Run the following cell to load the packages and dependencies that are going to be useful for your journey!

