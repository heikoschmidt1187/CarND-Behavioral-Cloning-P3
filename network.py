import csv
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Input, GlobalAveragePooling2D, Dropout

# read the csv file and store the data
lines = []

# open the csv file
with open('./training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    # read lines and add to list
    for line in reader:
        lines.append(line)

# extract center image and steering angle measurement
images = []
measurements = []

for line in lines:

    # get left and right images as well as centered
    for i in range(3):
        # get the filename of the center image and append to the current image path
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './training_data/IMG/' + filename

        # use OpenCV to read the image
        image = cv2.imread(current_path)
        images.append(image)

        # extract steering measurements
        # TODO: more elegant implementation
        if i == 0:
            measurements.append(float(line[3]))
        elif i == 1:
            measurements.append(float(line[3]) + 0.2)
        else:
            measurements.append(float(line[3]) - 0.2)

### image augmentation to make a greater and homogeneous training set
augmented_images = []
augmented_measurements = []

# loop through all images + measurements to augment the data
for image, measurement in zip(images, measurements):
    # append original data
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

### neural network and training ###
# convert to numpy arrays as this format is needed by Keras
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# use NVIDIA proved CNN --> https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()

# input and normalization layer 3@320x160
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# crop the top 70px and the bottom 25px to eliminate useless image information (trees, mountains, sky, hood)
# Output 3@320x65
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# First convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))

# Second convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))

# Third convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))

# Fourth convolutional layer, no stride, 3x3 kernel
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Fifth convolutional layer, no stride, 3x3 kernel
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Flatten Layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(100))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1))

# print model to see layers
for layer in model.layers:
    print(layer.output_shape)


# compile model, use mean square error loss function as it's a continuous output with regression
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

# save the model for usage
model.save('model.h5')
