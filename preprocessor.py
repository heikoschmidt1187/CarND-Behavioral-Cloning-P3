import csv
import random
import cv2
import numpy as np
import sklearn

class Preprocessor():
    def __init__(self, path):
        self.path = path
        self.samples = self.read_csv(self.path + 'driving_log.csv')
        self.max_y_shift = 40
        self.max_x_shift = 40
        self.image_cols = 320
        self.image_rows = 160

    def read_csv(self, path):
        """
        `path` Input path to csv file

        Reads line from a driving_log.csv file and returns all samples

        Returns samples from csv file
        """
        # open the csv file
        ret_samples = []

        with open(path) as csvfile:
            reader = csv.reader(csvfile)

            # read lines and add to list
            for line in reader:
                ret_samples.append(line)

        return ret_samples

    def generator(self, gen_samples, batch_size=32):
        """
        `gen_samples` Input samples to operate generator on
        `batch_size` Input batch size for generator handling

        Creates a generator to be used while training model.

        Returns shuffled batches for learning
        """
        # evaluate the number of samples for futher processing
        num_samples = len(gen_samples)

        # generators run forever
        while True:

            # shuffle samples to equalize distribution for learning
            random.shuffle(self.samples)

            # get offsets to go through the samples in defined batch_size stepts
            for offset in range(0, num_samples, batch_size):
                # get the current batch from samples
                batch_samples = self.samples[offset:offset + batch_size]

                # extract the images and measurements from the batches
                images = []
                measurements = []

                # loop through the samples for CPU preprocessing and augmentation
                for batch_sample in batch_samples:

                    for side in ['center', 'left', 'right']:
                        # read the center image
                        rgb, yuv, hls, hsv, measure = self.get_image_from_file(side, batch_sample)

                        # append the non augmented image and measurement
                        images.append(yuv)
                        measurements.append(measure)

                        # augment image: flip to simulate driving the track the outher
                        # way around, so curvatures get inverted --> avoid bias to one
                        # side as tracks in normal direction tends to bias on left curvatures
                        fl_img, fl_measure = self.flip_image(yuv, measure)

                        images.append(fl_img)
                        measurements.append(fl_measure)

                        # augment image: shift the image horizontally and vertically to
                        # simulate different lane positions
                        shift_img, shift_measure = self.shift_image(yuv, measure)

                        images.append(shift_img)
                        measurements.append(shift_measure)

                        # augment image: simulate different lighting conditions to
                        # teach the network to handle different daylight and abmient
                        # light conditions
                        bright_img = self.adapt_brightness(hsv)

                        images.append(bright_img)
                        measurements.append(measure)

                        # augment image: simulate random shadows as car seems to
                        # want to avoid them and is moving out of the road
                        shad_img = self.adapt_shadow(hls)

                        images.append(shad_img)
                        measurements.append(measure)

                # convert to numpy arrays as this format is needed by Keras
                X_train = np.array(images)
                y_train = np.array(measurements)

                yield sklearn.utils.shuffle(X_train, y_train)

    def adapt_shadow(self, hls):
        """
        `hls` Input image in hls format

        Does a random shadow augmentation to the l channel of the input image

        Returns the augmented image in YUV format
        """
        # get a random position at the top and the bottom of the input image
        top_y = self.image_cols * np.random.uniform()
        bot_y = self.image_cols * np.random.uniform()

        # build a mask to overlay later to simulate shadow using a mesh grid
        shad_mask = hls[:,:,1] * 0
        X_m = np.mgrid[0:hls.shape[0],0:hls.shape[1]][0]
        Y_m = np.mgrid[0:hls.shape[0],0:hls.shape[1]][1]
        shad_mask[(X_m * (bot_y - top_y) - self.image_rows * (Y_m - top_y) >= 0)] = 1

        # augment only part of the images randomly
        img_shad = np.copy(hls)

        if np.random.randint(2) == 1:
            rand_bright = 0.5

            # from the shadow mask, randomly draw the activated or not activated
            # pixels to get more variance
            rand = np.random.randint(2)
            img_shad[:,:,1][shad_mask == rand] = img_shad[:,:,1][shad_mask == rand] * rand_bright

        return cv2.cvtColor(cv2.cvtColor(img_shad, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2YUV)

    def adapt_brightness(self, hsv):
        """
        `rgb` Input image in hsv format

        Does a random brightness adaption on the value channel of the input image

        Returns augmented image in yuv colorspace
        """
        # get floating point numpy array
        bright_img = np.array(hsv, dtype=np.float64)

        # calculate a random brightness, half it as image should only be partly
        # augmented
        rand_bright = 0.5 * np.random.uniform()

        # augment the v channel
        bright_img[:,:,2] = bright_img[:,:,2] * rand_bright

        # truncate the values back to 255 to fit into desired uint8
        bright_img[:,:,2][bright_img[:,:,2] > 255] = 255

        # convert type back
        bright_img = np.array(bright_img, dtype=np.uint8)

        # return YUV image
        return cv2.cvtColor(cv2.cvtColor(bright_img, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2YUV)


    def shift_image(self, img, measure):
        """
        `img` Input image to be shifted
        `measure` Input measurement for shifted image

        Shifts the image horizontally and vertically, adapts measurement

        Returns shifted image and measurement
        """
        # get a random pixel value for x and y transition
        x_trans = self.max_x_shift * np.random.uniform() - self.max_x_shift/2
        y_trans = self.max_y_shift * np.random.uniform() - self.max_y_shift/2

        # build a translation matrix and do a affine warp to the image
        trans_matrix = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
        shift_img = cv2.warpAffine(img, trans_matrix, (self.image_cols, self.image_rows))

        # adapt measurement according to the shifted horizontal pixels
        # HINT: 0.004 steering angle per pixel shift works fine
        shift_measure = measure + x_trans / self.max_x_shift * 2 * 0.2

        return shift_img, shift_measure


    def flip_image(self, img, measure):
        """
        `img` Input image to be flipped
        `measure` Input measurement for flipped image

        Flips the image horizontally, adapts the measurement

        Returns flipped image and measurement
        """
        # flip the image
        ret_img = cv2.flip(img, 1)

        # invert measurement
        ret_measure = -1.0 * measure

        return ret_img, ret_measure

    def get_image_from_file(self, type, sample):
        """
        `type` Input type of image to get, can be 'left', 'center', 'right'

        This function loads the image corresponding to the given type.

        Returns the image and measurement for the given type of image in YUV, RGB
        and HSL colorspace
        """

        # get the correct image path index for csv read depending on image type
        if type is 'center':
            index = 0
        elif type is 'left':
            index = 1
        else:
            index = 2

        # get the current filename
        filename = sample[index].split('/')[-1]
        current_path = self.path + 'IMG/' + filename

        # read the image and convert it to different colospaces for further preprocessing
        img = cv2.imread(current_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # get the measurement according to the image type
        measurement = float(sample[3])

        if type is 'left':
            measurement = measurement + 0.2
        else:
            measurement = measurement - 0.2


        # return images and measurement
        return img_rgb, img_yuv, img_hls, img_hsv, measurement
