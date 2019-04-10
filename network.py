import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

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
    # get the filename of the center image and append to the current image path
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './training_data/IMG/' + filename

    # use OpenCV to read the image
    image = cv2.imread(current_path)
    images.append(image)

    # extract steering measurements
    measurements.append(float(line[3]))

### neural network and training ###
# convert to numpy arrays as this format is needed by Keras
X_train = np.array(images)
y_train = np.array(measurements)

# create a model
model = Sequential()

# add flatten imput layer
model.add(Flatten(input_shape=(160, 320, 3)))

# add dense output layer
model.add(Dense(1))

# compile model, use mean square error loss function as it's a continuous output with regression
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

# save the model for usage
model.save('model.h5')
