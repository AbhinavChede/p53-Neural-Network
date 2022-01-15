# p53-Neural-Network

Introduction:
This neural network is a fully connected ReLU network with a LogSoftMax final layer. This neural network classifies incoming structural data (type float) of the p53 gene into the current state of the gene (whether it is active or inactive).

Imports necessary:
numpy, pandas, torch, and transforms from torchvision

Dataset:
The dataset I used was the p53 Mutants Datset from the UCI Machine learning repository.
Link to dataset: https://archive.ics.uci.edu/ml/datasets/p53+Mutants

The dataset was not in the right form when I downloaded it so I had to do transform and edit the values.

IMPORTANT: The datset had some missing values, and I manually deleted them.

The dataset came in a .data file and I had to convert it to a .csv file

The dataset had around 17,000 instances and each instance had 5409 attributes (The last one is the label).
I divided the datset into a 70/30 ratio for the training and testing dataset.

In Neural Network:
The structure of the neural network is 5408 input size, 2 hidden layers of size 256, 16 and a output size of 2.
The loss I used was the NLLLoss and the optimizer was an SGD.

Learning rate - 0.0002
Momentum - 0.9
Accuracy - 99.24%

Notes:
At first, when I completed my neural network I had a lot of issues: overfitting, not random sampling, and other errors.
In order to fix them and make the data better:
    - Normlaize the data
    - Transform the data into a useful form
    - Replace zeroes with small decimals
    - Reduce and figure out an ideal structure for my neural network
    - Try out different activation functions to see which one works best
    - Implement different hyperparameters like learning rate and Momentum
    - and many more




