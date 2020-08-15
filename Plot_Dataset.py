from matplotlib import pyplot
from matplotlib.image import imread

#define loation of dataset
folder = 'dogs-vs-cats_dataset/train/train/'

#plot first few images
for i in range(9):
    #define subplot
    pyplot.subplot(330 + 1 + i)
    #define filename
    filename = folder + 'dog.' + str(i) + '.jpg'
    #load image pixels
    image = imread(filename)
    #plot raw pixel data
    pyplot.imshow(image)

pyplot.show()