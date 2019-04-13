import csv
import cv2
import numpy as np
import tensorflow as tf
import random
import math

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Input, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

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

def get_callbacks():
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

    return [earlystopping]

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
                    source_path = batch_sample[i]
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
                        measurements.append(float(batch_sample[3]))
                    elif i == 1:
                        measurements.append(float(batch_sample[3]) + 0.2)
                    else:
                        measurements.append(float(batch_sample[3]) - 0.2)

                    # flip images to simulate track other way around
                    images.append(cv2.flip(image, 1))

                    if i == 0:
                        measurements.append(float(batch_sample[3]) * -1.0)
                    elif i == 1:
                        measurements.append((float(batch_sample[3]) + 0.2) * -1.0)
                    else:
                        measurements.append((float(batch_sample[3]) - 0.2) * -1.0)

                    # random brightness adjustments to simulate different lighting conditions
                    image_brightness = cv2.cvtColor(image, cv2.COLOR_YUV2RGB);
                    image_brightness = cv2.cvtColor(image, cv2.COLOR_RGB2HSV);
                    image_brightness = np.array(image_brightness, dtype=np.float64)
                    rand_brightness = 0.5 * np.random.uniform()
                    image_brightness[:,:,2] = image_brightness[:,:,2] * rand_brightness
                    image_brightness[:,:,2][image_brightness[:,:,2] > 255] = 255
                    image_brightness = np.array(image_brightness, dtype=np.uint8)
                    image_brightness = cv2.cvtColor(image_brightness, cv2.COLOR_HSV2RGB)
                    image_brightness = cv2.cvtColor(image_brightness, cv2.COLOR_RGB2YUV)

                    images.append(image_brightness)

                    if i == 0:
                        measurements.append(float(batch_sample[3]))
                    elif i == 1:
                        measurements.append(float(batch_sample[3]) + 0.2)
                    else:
                        measurements.append(float(batch_sample[3]) - 0.2)

                    # shift images for lane switches
                    x_translation = 40 * np.random.uniform() - 40/2

                    if i == 0:
                        steering = float(batch_sample[3])
                    elif i == 1:
                        steering = float(batch_sample[3]) + 0.2
                    else:
                        steering = float(batch_sample[3]) - 0.2

                    steering_ang = steering + x_translation / 20*2*0.2

                    y_translation = 40 * np.random.uniform()- 40/2
                    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
                    trans_img = cv2.warpAffine(image, translation_matrix, (320, 160))

                    images.append(trans_img)
                    measurements.append(steering_ang)

                    # simulate shadows as they seem to make problems currently
                    top_y = 320*np.random.uniform()
                    top_x = 0
                    bot_x = 160
                    bot_y = 320*np.random.uniform()
                    image_rgb = cv2.cvtColor(image,cv2.COLOR_YUV2RGB)
                    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
                    shadow_mask = 0*image_hls[:,:,1]
                    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
                    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
                    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
                    #random_bright = .25+.7*np.random.uniform()
                    if np.random.randint(2)==1:
                        random_bright = .5
                        cond1 = shadow_mask==1
                        cond0 = shadow_mask==0
                        if np.random.randint(2)==1:
                            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
                        else:
                            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
                    image_hls = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
                    image = cv2.cvtColor(image_hls,cv2.COLOR_RGB2YUV)

                    images.append(image)

                    if i == 0:
                        measurements.append(float(batch_sample[3]))
                    elif i == 1:
                        measurements.append(float(batch_sample[3]) + 0.2)
                    else:
                        measurements.append(float(batch_sample[3]) - 0.2)


            # convert to numpy arrays as this format is needed by Keras
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

### neural network and training ###
# size for batch processing
batch_size = 16

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
model.add(BatchNormalization());

# Second convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
model.add(BatchNormalization());

# Third convolutional layer, 2x2 stride, 5x5 kernel
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
model.add(BatchNormalization());

# Fourth convolutional layer, no stride, 3x3 kernel
model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
model.add(BatchNormalization());

# Fifth convolutional layer, no stride, 3x3 kernel
model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
model.add(BatchNormalization());

# Flatten Layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))

model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))

# Output layer
model.add(Dense(1, activation='elu'))

# print model summary
model.summary()


# compile model, use mean square error loss function as it's a continuous output with regression
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / batch_size), \
    validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples) / batch_size),
    epochs=10, verbose=1, callbacks=get_callbacks())

# save the model for usage
model.save('model.h5')
