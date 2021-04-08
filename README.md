# Neural Network From Scratch

The Multiclass Perceptron Neural Network is developed from Scrach without using any Machine Learning libraries.

### Data
We use MNIST dataset of Handwritten digit images normalized into 28x28 (784) bit vectors.<br/>
The training set contains 60,000 images. Test set contains 10,000 images.

### Model Description

The Model is a Perceptron Neural Network with 1 Hidden Layer. 
* Input Layer - 784 Nodes
* Hidden Layer - 256 Nodes with Sigmoid as the activation function.
* Output Layer - 10 Nodes with Softmax as activation function.

### Training
The model is fine trained and fine-tuned using Stochastic Gradient decent.
The loss is computed using Cross Entropy loss function.
* Number of Epochs = 20
* Learning Rate = 0.005

### Results
The model achieves a training accuracy of **94.7%** and a test accuracy of **93.4%**. 


 
