# neural-net

## Author
Eli Boyden, aqu4eb, eboyden42
Completed while following course "Create a Neural Network in Java" by John Purcell, added additional functionality afterwards
https://www.udemy.com/course/neural-network-java

## To Run

1) Clone the repository and navigate to App.java.
2) Run the app and follow directions in the terminal to either create a new network or use an existing one. New networks 
will automatically train and then generate images and predictions.
4) For existing networks you can decide if you want to train the network further or continue 
to image and prediction generation.
5) Images will be generated in the root directory as "network_nameX.jpg" where X is a number from 1 to 11. If you 
generate images for the same network more than once the new images will be called "network_nameX|Y.jpg" where Y is 
a number to ensure no overwriting.
7) Images contain montages of handwritten digits where red digits were predicted incorrectly by the network.
8) The labelsX.txt files contain a table of numbers representing the true value of the handwritten digit 
corresponding to their location in each image.
9) The predictionsX.txt files contain a table of numbers representing the network predicted value of the handwritten digit
corresponding to their location in each image. These files are updated each time images are created and reflect the most recent network
run. The numbers that the network predicted incorrectly will have a "[" in front of them.

## Issues

If network sizes become too large then the Loss may appear as NaN. This is not an issue in most cases, however,
it is probably wise to end the program and try again with a smaller network.

## Extra Settings

Lines 60 to 65 contain variables that act as training settings, and you may change them at your discretion.
The two most useful of these will be number of epochs, and the learning rate variables