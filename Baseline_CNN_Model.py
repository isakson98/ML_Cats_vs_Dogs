# 
# VGG models -> deep convolutional networks
# stacking convolutinoal layers with small 3 x 3 filters followed by a max pollin layer
# these layers form a block, where the blocks are repeated with increased # of filters as 
# 32,64, 128, 256 for first 4 blocks of the model
# each layer with use ReLU activation function and He weight init (best practices)
# 
# 
# 

# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# what is max pooling and filters?
# reLU actiavtion function
# He wight initilization
# momentum in stochaitsic gradient descent
# binary cross-entropy loss function
# epochs?
# cross entroy loss -> train usually lower than test
#                      train and test both high -> underfit
#                      train and test diverge significantly -> overfit (to the train data)

# define cnn model
def define_model_1_block():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# explaning here
def define_model_2_blocks():
    model = Sequential()
    # we have convoluted layer
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    # followed by pooling in each layer
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # prior to 2nd to last layer, we flatten
    model.add(Flatten())
    # 2nd to last layer
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # applying sigmoid function in the last layer
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# define cnn model
def define_model_3_blocks():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation = 'relu', kernel_intialiazer='he_uniform', padding='same', input_shape=(200,200,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_intialiazer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves
#  model is overfitting, underfitting, or has a good fit...
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

#run the test harness for evaluating a model
def run_test_harness():
    #define model
    model = define_model_1_block()
    #create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    #prepare iterators
    # fit the model using train folder
    train_it = datagen.flow_from_directory('preprocessed_dogs-vs-cats_dataset/train/',
        class_mode='binary', batch_size=64, target_size = (200,200))
    # test_it will act as a cross-validation dataset ( as smaller portion of earlier divided train folder)
    test_it = datagen.flow_from_directory('preprocessed_dogs-vs-cats_dataset/test/',
        class_mode='binary', batch_size=64, target_size = (200,200))
    # fit model 
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
    #evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc & 100.0))
    #learning curves
    summarize_diagnostics(history)

run_test_harness()



