import csv
import cv2
import numpy as np
import tensorflow as tf
import random
import math

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Input, GlobalAveragePooling2D, Dropout
from keras import optimizers

import sklearn
from sklearn.model_selection import train_test_split

# create samples to hold the paths
samples = []

# open the csv file
with open('./training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    # read lines and add to list
    for line in reader:
        samples.append(line)

# split samples into training and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# generator to handle batches
def generator(samples, batch_size=32):
    num_samples = len(samples)

    # generators run forever
    while True:
        random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            # extract center image and steering angle measurement
            images = []
            measurements = []

            for batch_sample in batch_samples:

                # get left and right images as well as centered
                for i in range(3):
                    # get the filename of the center image and append to the current image path
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = './training_data/IMG/' + filename

                    # use OpenCV to read the image
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)

                    # extract steering measurements
                    # augmentation
                    # TODO: more elegant implementation
                    if i == 0:
                        measurements.append(float(line[3]))
                    elif i == 1:
                        measurements.append(float(line[3]) + 0.2)
                    else:
                        measurements.append(float(line[3]) - 0.2)

                    images.append(cv2.flip(image, 1))

                    if i == 0:
                        measurements.append(float(line[3]) * -1.0)
                    elif i == 1:
                        measurements.append((float(line[3]) + 0.2) * -1.0)
                    else:
                        measurements.append((float(line[3]) - 0.2) * -1.0)


            # convert to numpy arrays as this format is needed by Keras
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

### neural network and training ###
# size for batch processing
batch_size = 32

# generators for training and validation
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# use NVIDIA proved CNN --> https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()

# input and normalization layer 3@320x160
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# crop the top 70px and the bottom 25px to eliminate useless image information (trees, mountains, sky, hood)
# Output 3@320x65
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# First convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))

# Second convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))

# Third convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))

# Fourth convolutional layer, no stride, 3x3 kernel
model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))

# Fifth convolutional layer, no stride, 3x3 kernel
model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))

# Flatten Layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(100, activation='elu'))
#model.add(Dropout(0.2))

model.add(Dense(50, activation='elu'))
#model.add(Dropout(0.5))

model.add(Dense(10, activation='elu'))

# Output layer
model.add(Dense(1, activation='elu'))

# print model to see layers
for layer in model.layers:
    print(layer.output_shape)


# compile model, use mean square error loss function as it's a continuous output with regression
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / batch_size), \
    validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples) / batch_size),
    epochs=5, verbose=1)

# save the model for usage
model.save('model.h5')
