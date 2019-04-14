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

To avoid overfitting, I introduced a Keras Dropout layer with a droput rate of 0.5. The final architecture looks like this:

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
  conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
  _________________________________________________________________
  conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
  _________________________________________________________________
  conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
  _________________________________________________________________
  conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
  _________________________________________________________________
  flatten_1 (Flatten)          (None, 2112)              0
  _________________________________________________________________
  dense_1 (Dense)              (None, 100)               211300
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 100)               0
  _________________________________________________________________
  dense_2 (Dense)              (None, 50)                5050
  _________________________________________________________________
  dense_3 (Dense)              (None, 10)                510
  _________________________________________________________________
  dense_4 (Dense)              (None, 1)                 11
  =================================================================
  Total params: 348,219
  Trainable params: 348,219
  Non-trainable params: 0
```

The five convolution layers are designed to perform feature extraction according to the NVIDIA article on the network. They have been determined through experiments. The first three convolutions are strided with 2x2 and a 5x5 kernel. The other two convolution layers are non-strided and use a kernel size of 3x3.

The convolution layers are followed by a Keras Flatten layer to build one huge vector of values, that then is fed into three fully connected layers. The last layer is the output layer of the steering angle.

In the next chapters, you will find more detailed descriptions on the architecture and why it was chosen. You can find the model implementation in *nvidiamodel.py*, function *generate_model()* lines 26 to 76.


#### 2. Attempts to reduce overfitting in the model

I was really suprised that with NVIDIA's model, even though its simplicity, I didn't much problems with overfitting. I played around with several Dropout layers with different dropout rates at different positions, but in the end there wasn't really an improvement to the performance in the autonomous track. I also tested BatchNormalization layers after the convolutions, but neither did they improve the learing time, nor did they improve the overall performance.

So in the end I left the model with one dropout layer after the first fully connected layer and a dropout rate of 0.5 (nvidiamodel.py, line 65).

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

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
