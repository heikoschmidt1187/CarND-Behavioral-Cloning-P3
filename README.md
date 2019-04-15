# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[cnn_architecture]: ./images/cnn_architecture.png "NVIDIA CNN architecture"
[center_driving_image]: ./images/center_2019_04_13_17_30_11_643.jpg "Center lane driving"
[left_cam]: ./images/left_2019_04_13_20_29_46_796.jpg "Center lane driving"
[center_cam]: ./images/center_2019_04_13_20_29_46_796.jpg "Center lane driving"
[right_cam]: ./images/right_2019_04_13_20_29_46_796.jpg "Center lane driving"
[initial_distribution]: ./images/01_training_data.png "Initial distribution"
[3cam_distribution]: ./images/02_trainging_data_3cams.png "Three cam distribution"
[03_augmented]: ./images/03_augmented_data.png "Three cam distribution"
[04_augmented]: ./images/04_augmented_data.png "Three cam distribution"
[05_rgb]: ./images/05_rgb.png "RGB image"
[05_images]: ./images/05_images.png "Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model.py** containing the script to create and train the model. It relies on
  - **nvidiamodel.py**, which implements a class for the used CNN model
  - **preprocessor.py**, which contains the code for image augmentation for the learning and validation process
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network
* **output_video.mp4** containing a successful autonomous run of track 1 in the simulator
* **README.md** (this document) containing the writeup summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
  python drive.py model.h5
```

The drive.py file has been adapted to have a target speed or 20 mph and to convert the camera images to YUV colorspace which is used by the CNN.

#### 3. Submission code is usable and readable

For any futher understanding of the mentioned code file, refer to the comments directly in the source code. The following description is a brief overview over the delivered files.

The *model.py* file contains the integration of the preprocessor and the model. It initializes the preprocessor and the model with the correct path to the learning dataset, meaning the images from the simulator and the drive_log.csv with imagepaths and measurement data. Additionally, some important hyperparameters for training the model can be set here.

The file *preprocessor.py* contains code for loading the training data from the driving_log.csv files and reading the training images. It uses a generator to minimize the memory footprint on training the network through using batches on the augmented training data. The Preprocessor class also contains helper functions for the image augmentation described later.

In file *nvidiamodel.py* you can find the class NvidiaModel which implements the CNN based on Keras, used in this project. As one can guess from the name, it uses the [NVIDIA End-to-End Deep Learning model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) architecture. Through the object oriented approach, the CNN can easily be replaced by a new class and - if needed - another image preprocessor.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The convolution neural network I used is based on the [NVIDIA End-to-End Deep Learning model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The following image shows the original architecture:

![NVIDIA CNN architecture][cnn_architecture]

I adapted the model's input layer to take the images of the format 320x160px to fit the images we retrieve from the camera in the simulated car. The images are converted to YUV colorspace like the ones in the NVIDIA article.

For the output layer I chose a Kerase Dense layer with 1 value as output - the steering angle.

In addition to that, I used a Keras Lambda layer as input layer for normalization, and a Cropping2D layer to remove parts of the image that are not interesting for the model.

To avoid overfitting, I introduced a Keras BatchNormalization layer after each Convolution and Fully Connected layer. The resulting model looks like this:

```sh
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  lambda_1 (Lambda)            (None, 160, 320, 3)       0
  _________________________________________________________________
  cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
  _________________________________________________________________
  batch_normalization_1 (Batch (None, 31, 158, 24)       96
  _________________________________________________________________
  activation_1 (Activation)    (None, 31, 158, 24)       0
  _________________________________________________________________
  conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
  _________________________________________________________________
  batch_normalization_2 (Batch (None, 14, 77, 36)        144
  _________________________________________________________________
  activation_2 (Activation)    (None, 14, 77, 36)        0
  _________________________________________________________________
  conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
  _________________________________________________________________
  batch_normalization_3 (Batch (None, 5, 37, 48)         192
  _________________________________________________________________
  activation_3 (Activation)    (None, 5, 37, 48)         0
  _________________________________________________________________
  conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
  _________________________________________________________________
  batch_normalization_4 (Batch (None, 3, 35, 64)         256
  _________________________________________________________________
  activation_4 (Activation)    (None, 3, 35, 64)         0
  _________________________________________________________________
  conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
  _________________________________________________________________
  batch_normalization_5 (Batch (None, 1, 33, 64)         256
  _________________________________________________________________
  activation_5 (Activation)    (None, 1, 33, 64)         0
  _________________________________________________________________
  flatten_1 (Flatten)          (None, 2112)              0
  _________________________________________________________________
  dense_1 (Dense)              (None, 100)               211300
  _________________________________________________________________
  batch_normalization_6 (Batch (None, 100)               400
  _________________________________________________________________
  activation_6 (Activation)    (None, 100)               0
  _________________________________________________________________
  dense_2 (Dense)              (None, 50)                5050
  _________________________________________________________________
  batch_normalization_7 (Batch (None, 50)                200
  _________________________________________________________________
  activation_7 (Activation)    (None, 50)                0
  _________________________________________________________________
  dense_3 (Dense)              (None, 10)                510
  _________________________________________________________________
  batch_normalization_8 (Batch (None, 10)                40
  _________________________________________________________________
  activation_8 (Activation)    (None, 10)                0
  _________________________________________________________________
  dense_4 (Dense)              (None, 1)                 11
  =================================================================
  Total params: 349,803
  Trainable params: 349,011
  Non-trainable params: 792
```

The five convolution layers are designed to perform feature extraction according to the NVIDIA article on the network. They have been determined through experiments. The first three convolutions are strided with 2x2 and a 5x5 kernel. The other two convolution layers are non-strided and use a kernel size of 3x3.

The convolution layers are followed by a Keras Flatten layer to build one huge vector of values, that then is fed into three fully connected layers. The last layer is the output layer of the steering angle.

In the next chapters, you will find more detailed descriptions on the architecture and why it was chosen. You can find the model implementation in *nvidiamodel.py*, function *generate_model()* lines 26 to 76.


#### 2. Attempts to reduce overfitting in the model

I was really suprised that with NVIDIA's model, even though its simplicity, I didn't much problems with overfitting. I played around with several Dropout layers with different dropout rates at different positions, but in the end there wasn't really an improvement to the performance in the autonomous track.

So the first semi-final model had a dropout layer after the first fully connected layer with a dropout rate of 0.5. The car did drive multiple rounds autonomous on track 1. While this was enough to meet the project requirements, I read an interesting paper about [BatchNormalization - Sergey Ioffe, Christian Szegedy](https://arxiv.org/abs/1502.03167). So I removed the dropout layer and introduced multiple BatchNormalization layers (lines 31 to 69 in nvidiamodel.py), and this improved not only the learning time, but also the performance on the track. While the car is still shaky, it can drive around the tracks for hours.

In order to monitor the learning progress, I used a shuffled subset of the training data as a validation set (nvidiamodel.py, line 90). The validation size is 20% of the training data.

While training the model, I watched the loss of the training data and the validation data. Both should decrease in a monotonic way. To stop the process if the loss isn't decreasing anymore, I used a simple EarlyStopping callback - *nvidiamodel.py*, function *get_callbacks()*, line 18 to 24:

```python
  def get_callbacks(self):
      """
      Returns the callbacks during model fit
      """
      earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

      return [earlystopping]
```

Instead of using a dedicated testset which should only used once to verify the training, I simply did run the model with the autonomous mode of the simulator to test if it can drive around the track and how it drives.

#### 3. Model parameter tuning

The model uses the Adam optimizer with a learning rate of 0.001. During my tests I lowered the learning rate to 0.0005 and 0.0001, but this didn't improve the performance and the learning of the CNN took much more epochs, so I set it back to the 0.001 default value.

#### 4. Appropriate training data

Udacity provides a sample training set of data, but I wanted to use my own dataset. So I collected it the following way:

* Three to four rounds driving in the center of the road as good as possible in one direction
* To avoid bias to the left, I drove one round in the opposite direction
* As the car had problems with the texture on the bridge, I collected some good driving examples dedicated to that case
* Because the car had problems in turnings with red/white road boarders, I drove additional data on that
* I avoided collecting recovery from bad positions, as with previous tests this lead to bad autonomous driving behavior --> I solved this with image augmentation described later.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At the very beginning, I started implementing and using the LeNet architecture for the first learning processes. I collected two rounds of driving mostly in the centere of the road, only using the center camera image for training. While I ensured that the training process was working and the toolchain was ok, the driving was terrible. The car instantly turned to the left and up the hills.

To see how well the model is working, I split the learning dataset into a training and a validation set. While learning, I watched the progression of the loss (mean squared error) to keep monotonic getting smaller and checked how the model is doing on the validation set in each learning epoch.

Soon I realized, that the LeNet architecture was not enough and I started researching for specific alternatives on the topic. As the Udacity lesson put attention on transfer learning, I first thought maybe the GoogLeNet may be a good choice to start as it's good in image classification and fast through it's architecture. Then I found the NVIDIA model mentioned above and the comma.ai model for autonomous driving. As the NVIDIA model proved it's usability for autonomous driving in practice, I decided to use it as well with small modifications, also mentioned above.

So when I implemented the model, the car was directly able to drive one round autonomous on track 1 if it wasn't disturbed. When the car got too close to the edge of the road, it failed. Also the areas with water made problems.

As I've already written, I did not have too much problems with overfitting and introduced in the end BatchNormalization layers as they in some cases avoid the need of dropout layers and make the model less sensitive to the initialization of the parameters while doing multiple learning sessions.

To introduce more generalization, I augmented the training data, described in the chapters below.

I saw one interesting thing when doing the test drives on the track: I trained the network with images collected at smallest resolution and lowest detail level to save time. This led to the case that when driving with a full detailed version on high resolution, the car tried to avoid shadows on the road.

In order to face that issue, I augmented the data with random shadows and drove the specific parts in the simulator two times to collect training data with shadows.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for hours.


#### 2. Final Model Architecture

I've already shown the final model architecture (nvidiamodel.py, lines 26 to 88) in chapter 1 for Model Architecture and Training strategy above. You can read the layers and sizes there, it is a print of the model.summary() call of Keras.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on the track driving as accurate as possible in the middle of the lane, and used the center camera images for training the model. Below you can see an example of a center lane driving training image:

![Center lane driving][center_driving_image]

As I've already written, I did not directly created lane recovery data as this led my model to drive very badly. Instead of that, I first relied on the left and right camera images that are recorded as well. I used them with a steering angle offset of +/-0.2 with respect to the measured steering angle to introduce information on how to handle situations directly next to the centered lane data. This is done in *preprocessor.py*, line 66, and function get_image_from_file, lines 209 to 248.

Here are two examples for left, center and right camera images:

![left camera image][left_cam]
![center camera image][center_cam]
![right camera image][right_cam]

The initial dataset leads to the following distribution of steering angles:

![Initial steering angle distribution][initial_distribution]

As you can see, the center image data is heavily biased to either 0 steering angle while driving straight, and with negative angles as the track in the forward direction has mostly left curvatures.

The next image shows the data of all three images per sample with the corresponding measurement distribution:

![Steering angle distribution 3 cams][3cam_distribution]

It can be seen that the distribution is now more equally distributed with peaks at 0, 0.2 and -0.2 angles - looks like three Gaussians.


To generalize more, I augmented the image dataset a lot. This is all done in the *preprocessor.py* file, function *generator*, lines 36 onward. Please refer to that file for detailed information on the following descriptions.

Please refer to the article (An augmentation based deep neural network approach to learn human driving behavior)(https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) by Vivek Yadav as it was really helpful to me and supported me in implementing the following techniques.

First, I shuffle the samples before each epoch:

```python
  # shuffle samples to equalize distribution for learning
  random.shuffle(self.samples)
```

Then for each data sample, I load the center, left and right image file with the corresponding measurement, augmented with +0.2 for left and -0.2 for right:

```python
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
```

The image is also converted to different colorspaces for futher processing described shortly. Input to the model is the YUV image.

Next, the images and measurements are flipped to simulate driving the other way around in the track. So the intent is to teach for each left curvature a right curvature as well:

```python
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
```

Flipping images led to the following, even better distribution:

![Flipping augmented data][03_augmented]

After that, I want to simulate driving off the ideal line even more, by shifting the images with a affine warp to left and right. An additional up and down shift is done to simulate driving in hills:

```python
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
      shift_measure = measure + x_trans / self.max_x_shift * 2 * 0.004

      return shift_img, shift_measure
```

The shift is done in a random manner until max_x_shift pixels horizontally and max_y_shift pixels vertically. The current values are 40 pixels in each direction.

I also applied some random brightness adaption to simulate different lighting conditions and ambient lights:

```python
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
```

The adaption is done by manipulating the v channel of an hsv image in a random manner.

Last but not least, I tried to simulate different shadow distributions by randomly taking a top and bottom position and darken the l-channel of a hls image:

```python
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
```

With all the augmentation done, the final distribution looks the following:

![Final distribution][04_augmented]

As you can see, there are now a lot of peaks on left and right, showing the car how to steer when leaving the ideal lane. This is some kind of other extreme: while the first distributions had a peak on driving straight, this one has peaks on teaching how to react when leaving the center line. This leads to some kind of shaky driving sometimes and can be improved further by collecting more center driving images or reducing the augmentation a little.

To get an idea of the augmentation, see the following RGB image:

![RGB][05_rgb]

The random augmentation looks like this:

![Images][05_images]

After collecting and augmentation, I had 119,115 data points. The shuffled data is used 80% for training and 20% for validation.

The number of epochs for training turned out to be best when choosing 10. But as I wrote earlier, I used an EarlyStopping callback to stop learning when there's no further decrease in the validation loss. It turned out that the ideal number of epoches is around 8-10.
