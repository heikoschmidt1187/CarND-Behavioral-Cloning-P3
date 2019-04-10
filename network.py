import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D

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

# Own LeNet for test

# create a model
model = Sequential()

# preprocess image befor processing in the network
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# First convolutional layer
model.add(Convolution2D(6, 5, 5, activation='relu'))

# First Max Pooling Layer
model.add(MaxPooling2D())

# Second convolutional layer
model.add(Convolution2D(6, 5, 5, activation='relu'))

# Second Max Pooling Layer
model.add(MaxPooling2D())

# Flatten layer
model.add(Flatten())

# fully connected layers
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# compile model, use mean square error loss function as it's a continuous output with regression
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

# save the model for usage
model.save('model.h5')
