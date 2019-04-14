from preprocessor import Preprocessor

import math

from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Input, Dropout, Conv2D
from keras.callbacks import EarlyStopping


class NvidiaModel():
    def __init__(self, preprocessor):
        self.model = self.generate_model()
        self.preprocessor = preprocessor

    def get_callbacks(self):
        """
        Returns the callbacks during model fit
        """
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

        return [earlystopping]

    def generate_model(self):
        """
        Generates the NVIDIA model from Keras (through Tensorflow)
        """
        # use NVIDIA proved CNN --> https://devblogs.nvidia.com/deep-learning-self-driving-cars/
        model = Sequential()

        # input and normalization layer 3@320x160
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

        # crop the top 70px and the bottom 25px to eliminate useless image information (trees, mountains, sky, hood)
        # Output 3@320x65
        model.add(Cropping2D(cropping=((70, 25), (0, 0))))

        # First convolutional layer, 2x2 stride, 5x5 kernel
        model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        #model.add(BatchNormalization());

        # Second convolutional layer, 2x2 stride, 5x5 kernel
        model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        #model.add(BatchNormalization());

        # Third convolutional layer, 2x2 stride, 5x5 kernel
        model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        #model.add(BatchNormalization());

        # Fourth convolutional layer, no stride, 3x3 kernel
        model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
        #model.add(BatchNormalization());

        # Fifth convolutional layer, no stride, 3x3 kernel
        model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
        #model.add(BatchNormalization());

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

        return model

    def train_model(self, parameter_set, model_save_path='model.h5'):
        """
        `batch_size` Input Batch size for fit_generator training.
        `test_size` Input percentage of samples to use as validation set
        `learning_rate` Input learning rate for Adam optimizer
        `epochs` Input number of epochs to train model
        `model_save_path` Input path to file where trained model should be saved

        Trains the model through fit_generator based on the given bath_size. Use
        a good batch_size according to your GPU hardware to gain maximum performance.
        """
        # split samples into training and validation data
        train_samples, validation_samples = train_test_split(self.preprocessor.samples, test_size=parameter_set['validation_set'])

        # generators for training and validation
        train_generator = self.preprocessor.generator(train_samples, batch_size=parameter_set['batch_size'])
        validation_generator = self.preprocessor.generator(validation_samples, batch_size=parameter_set['batch_size'])

        # compile model, use mean square error loss function as it's a continuous output with regression
        adam = optimizers.Adam(lr=parameter_set['learning_rate'])

        # compile model with adam optimizer
        self.model.compile(loss='mse', optimizer=adam)

        # fit model with generators
        self.model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / parameter_set['batch_size']), \
            validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples) / parameter_set['batch_size']),
            epochs=parameter_set['epochs'], verbose=1, callbacks=self.get_callbacks())

        # save the model for usage
        self.model.save(model_save_path)
