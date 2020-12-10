# Resolution

The Resolution of the Images has a big impact on
the algorithm. The size of an image grows with 
the square of the dimensions of the image as seen in figure 1, and 
both the runtime memory consumption and disk
space that the project takes up are proportional
to the size of the images. Additionally, the
number of connections between neural network 
layers is proportional to the square of the size
of the image and thus the fourth power of the
dimension of the image.

Figure 1: Sizes of Images at different 
resolutions
![Sizes of different resolutions](Resources/Resolution%20Sizes.png)

Thus, It is imperative to use the smallest
image resolution possible in order both to minimize
memory footprint and maximize performance.
However, there is obviously a cost to using 
lower resolution images and that is the 
quality of the images. As the resolution
decreases, features of the image become more
blurred and difficult to discern. Figures 2 and
3 demonstrate this by putting a 160p image next
to a 360p image. It's easy to see that the 360p
image is much clearer and easier to discern
features from.

Figure 2: 160p image
![160p image](Resources/160p.png)

Figure 3: 360p image
![360p image](Resources/360p.png)

The question then becomes, what amount of detail
is necessary for the images to have? A good rule
of thumb about what resolution is best is 
whether a human can discern the desired features
in the image. If, for example, a human can't 
read text on an image, it's reasonable to assume
that a neural network would likewise not be able
to read that text.

This text example is exactly the metric used to
determine the resolution we chose to use. Since
the Player Identifier Neural Network gives a
text output, we ought to use a resolution that
allows us to read the relevant text in the
image. The Player Identifier network needs to
be able to read the names of the players from a
voting screen which is shown in figures 2 and 3.
It is clear to tell from observation that the
text in figure 2 is not discernible whereas that
in figure 3 is. Thus, we chose an image 
resolution of 360p for this project

# Data Labeling

Since all of our code will be supervised
learning activities, It is necessary for us to
label our data so that our models can look for
something to fit. Obtaining a Sufficient 
Quantity of useful data is difficult. In order
to make the process of labeling data easier
and more fun, we have a HTML program that will
display an image to a labeler and let them
classify it using buttons on the webpage and
have the image automatically stored after the
button press.


