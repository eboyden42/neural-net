# neural-net

## Overview
In this project I constructed and implemented a transformer neural network in Java without the use of machine learning
libraries. Networks created in this project (running from App.java) are trained on data from the MNIST handwritten 
digit dataset (https://www.tensorflow.org/datasets/catalog/mnist). The network's goal is to look at pixel data (784 input pixels)
and predict which number from 0 to 9 was written.

### Network Creation
To create a new network simply run the program, and enter a network name that doesn't already exist in the project directory.
If your name does not have a ".ntw" at the end, one will be appended. You can then choose to create a new network with
default settings, or specify the number of layers and nodes yourself.

### Training
Newly created networks, or existing ones if you choose, are trained on the MNIST training data for a default of 100 epochs
(this can be changed however, view Extra Settings below), where one epoch is one pass through the training data. Generally 
this will get you pretty good results, although as you try different sizes of networks you will find that some 
networks perform much worse than others, and networks that are too large may cause problems in the engine (see Issues). It
should go without saying that the larger networks will take longer to train than the smaller ones, so be prepared to wait
if you want to train large networks.

### Image Generation
After a network is created images will automatically be created in the root directory under the names "network_nameX.jpg" 
where X is a number from 1 to 11. If you generate images for the same network more than once the new images will be 
called "network_nameX.Y.jpg" where Y is a number to ensure no overwriting. These images show a series of handwritten
digits from MNIST, and colors the ones the network got wrong in red.

### Viewing Predictions
If you notice that your network got a digit wrong from the images and want to see what it guessed, you can! Simply navigate
to the predictionsX.txt file with the corresponding X value from your image. These show a grid with a 1 to 1 correspondence
with the images. Predictions files are updated automatically after each run to reflect the predictions of the latest run network.
I've added a "[" before each incorrect prediction in the predictions files.

### Viewing Labels
Sometimes even people can't really tell easily what these numbers are, so I've added some "labelsX.txt" which contain the
true value of each number from MNIST. Labels are formatted in a table similar to the predictions.

### Loading Preexisting Networks
After creating a network it will be saved in the root directory as "network_name.ntw". You can reuse these networks
by entering their name into the prompt when the program is run (with or without the .ntw). You may then choose if you want
to train the network, or simply continue to image, prediction and label generation.


## Author
Eli Boyden, aqu4eb, eboyden42
Completed while following course "Create a Neural Network in Java" by John Purcell, added additional functionality afterwards
https://www.udemy.com/course/neural-network-java

## To Run

1) Clone the repository and navigate to src/main/java/research/App.java.
2) Run the app and follow directions in the terminal to either create a new network or use an existing one. New networks 
will automatically train and then generate images and predictions.

## Issues

If network sizes become too large then the Loss may appear as NaN. This is not an issue in most cases, however,
it is probably wise to end the program and try again with a smaller network. You could try adjusting the scale initial weights
to a smaller value (Extra Settings), but caution is advised.

## Extra Settings

Lines 60 to 65 contain variables that act as training settings, and you may change them at your discretion.
The two most useful of these will be number of epochs, and the learning rate variables