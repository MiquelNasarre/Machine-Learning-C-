# Machine Learning C++

This project is my introduction to Machine Learning. In order to properly interiorize the math behind it 
I felt like the best way of introducing myself to the topic was to create a Machine Learning model from 
scratch.

As the learning task I chose the **MNIST** database, which is widely regarded as a good introduction to the 
subject and sounded like a fun project to undertake, since it meant teaching your computer to recognise 
handwritten digits reliably from scratch.

After many lines of code, I finally got to a point where I was satisfied with my knowledge gained from 
the dataset, having created a fully functional and flexible multi-layer Neural Network class that used 
SGD with momentum for training, and Cross Validation to choose the correct hyperparameters. And obtaining 
good results with a very simple layout.

This program is not by any means an attempt to build a practical Machine Learning library, the code does 
not use GPU for speed, it does not multithread or optimize matrix operations, therefore it is considerably 
slower than existing libraries like **torch**, **tensorflow** or **keras**. But it is a simple and robust 
implementation of the basics of Machine Learning and a good way of getting introduced to the topic.

## Requirements

- [Visual Studio](https://visualstudio.com): The program was designed using Visual Studio and files are linked
  through its infrastructure, therefore it is much simple to use it with Visual Studio. Nevertheless it is
  cross platform so any 64-bit machine should be able to run it after proper file linkage.

## Functionality

In order to showcase how to use the Neural Network class we will follow the steps found in the **main.cpp** 
file to train a simple Neural Network to recognise handwritten digits.

We can start by intializing the model, this is done by providing a layout (will initialize the weights randomly 
using He uniform for hidden layers) or by loading a previous model stored in a file:

```cpp
#define IMAGE_DIM 28 * 28

const unsigned layers[] = { IMAGE_DIM, 16, 16, 10 };
NeuralNetwork NN_train(4, layers);
```

```cpp
const char filename[] = "CV_3B1B";
const char nn_name[] = "96.25%";

NeuralNetwork NN_showcase(nn_name, filename);
```

Once the model is initialized we can load the training data and testing data, and feed the training data to the 
model. This is done using our dependencies made for loading the **MNIST**, in this case we store the values as 
simple **floats** in the range **\[0,1\]**. The images being **float\[784\]** and the labels being **unsigned**'s.

To feed the training images we can just use the function **NeuralNetwork::feed_data()** that will simply keep the 
image pointers inside the class for training.

```cpp
#include "NumberRecognition.h"

float* const* training_images = NumberRecognition::getImages(TRAINING, 0, 60000);
unsigned const* training_labels = NumberRecognition::getLabels(TRAINING, 0, 60000);

float* const* testing_images = NumberRecognition::getImages(TESTING, 0, 10000);
unsigned const* testing_labels = NumberRecognition::getLabels(TESTING, 0, 10000);

NN_train.feedData(60000, training_images, training_labels);
```

Finally we can set the hyperparameters and start a simple training loop using the train function. This function 
uses Stochastic Gradient Descent with momentum, decoupled Learning Rate with cosine decay and weight decay, 
therefore we can set all those values before training and call the **NeuralNetwork::trainWeights()** function.

```cpp
const unsigned epochs = 10;           // Since it is a small NN it does not need a lot of epochs
const float learning_rate = 0.0004f;  // Learning Rate must be small since it does not do batches
const float weight_decay = 0.0f;      // Due to the simplicity of the model WD is harmful
const float momentum = 0.85f;         // Strong momentum for averaging direction and fast convergence
const float lr_alpha = 0.01f;         // for decoupled LR we set the final LR to 1% of the initial

// We set the specified hyperparameters inside the model
NN_train.setLearningRate(learning_rate);
NN_train.setWeightDecay(weight_decay);
NN_train.setMomentum(momentum);
NN_train.setLRalpha(lr_alpha);

// We start the training loop for 10 epochs
NN_train.trainWeights(epochs);
```

After a simple training session like this we can expect good results with accuracy on the test set of around 
**~95.5%**. To be sure, we can calculate it and print it to the console, and save the weights before ending the process:

```cpp
float prediction_error = NN_train.computePredictionError(10000, testing_images, testing_labels);
float prediction_rate = NN_train.computePredictionRate(10000, testing_images, testing_labels);

printf("\nUpdated Prediction Error: %.4f", prediction_error);
printf("\nUpdated Prediction Rate:  %.2f%%", prediction_rate * 100.f);

const char nn_name[] = "trained weights";
NN_train.storeWeights(nn_name);
```

After running this simple program with some additional prints we obtain the following result: (non-deterministic)

```cmd
Neural Network correctly generated.

Initial Prediction Error: 2.3763
Initial Prediction Rate:  9.87%

Started training weights ...
Finished training in 51.9902s.

Updated Prediction Error: 0.1382
Updated Prediction Rate:  95.83%

Weights saved correctly with name "trained weights"
```

## Results

The layout I chose is as seen previously a simple dense Neural Network with two hidden layers of 16 nodes each. 
This design is inspired by the amazing introductory video series of [3Blue1Brown](https://youtu.be/aircAruvnKk) 
in Machine Learning. Although a simple layout, it managed to produce good results with proper tuning.

For the winning weights I used cross validation with randomized hyperparameters for 50 tries until it landed 
on a good combination and got **96.25%** accuracy. This set of weights is stored in the project folder with 
the name **CV_3B1B** and is the one that pops up when you run the project in ***showcase*** mode.

Inspired by the mentioned YouTube series I decided I also wanted to see the weights of the first layer as images 
so with the help of an old class, I managed to represent the 16 nodes of the first hidden layer as images, 
giving some, in my view, beautiful patterns of what the NN is actually recognising. 

You can see them displayed in the following collage:

---
<img width="1180" height="1180" alt="weights_collage" src="https://github.com/user-attachments/assets/35f30c39-510f-48b8-b973-f89cedd1a4d1" />

---
The choice for this model was the simplicity and the challenge to obtain good results with such a simple Neural Network. 
Of course using the same program I just showed with hidden layers of sizes **128** and **64** can produce more accurate 
models:

```cmd
Neural Network correctly generated.

Initial Prediction Error: 2.3316
Initial Prediction Rate:  14.42%

Started training weights ...
Finished training in 782.0298s.

Updated Prediction Error: 0.0642
Updated Prediction Rate:  98.47%

Weights saved correctly with name "trained weights"
```

But that defeats the purpose of the project and drowns it in computational optimizations instead of focusing on 
learning the basic concepts of Machine Learning.

## Future Plans

After achieving results I was happy with on the **MNIST** database I was ready to undertake more challenging projects, 
with the intention of getting to the **CIFAR-100**, but reality struck and I noticed that if I wanted to include 
Convolutional layers into my project and get the flexibility I wanted out of the Neural Networks I needed a complete 
rewrite. And to overcome the computational challenges I would need to implement GPU processing, and maybe my computer 
was not even powerful enough for such databases. 

I wrote in the file "Wishes for the future.h" a little sketch of how I wanted my library to operate, and it turns 
out that it is very similar to how other existing libraries already function.

So maybe in the future I'll decide to take the challenge but for the time being I think for my learning experience 
switching to **torch** and **python** is the correct decision. You can find my progress in the repository 
[Python Machine Learning](https://github.com/MiquelNasarre/Python-Machine-Learning.git).
