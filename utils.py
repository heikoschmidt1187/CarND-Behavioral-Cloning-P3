import csv
import cv2
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

preprocessor = Preprocessor('./training_data/')

# create samples to hold the paths
samples = []

# open the csv file
with open('./training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    # read lines and add to list
    for line in reader:
        samples.append(line)

angles = []

for sample in samples:
    angles.append(float(sample[3]))

plt.hist(angles, bins=40, align='mid')
plt.show()

# augmentation

angles = []

for sample in samples:
    angles.append(float(sample[3]))
    angles.append(float(sample[3]) + 0.2)
    angles.append(float(sample[3]) - 0.2)
    angles.append(-1.0 * float(sample[3]))
    angles.append(-1.0 * (float(sample[3]) + 0.2))
    angles.append(-1.0 * (float(sample[3]) - 0.2))


plt.hist(angles, bins=40, align='mid')
plt.show()


# productive Augmentation
images = []
measurements = []


for sample in preprocessor.samples:

    for side in ['center', 'left', 'right']:
        # read the center image
        rgb, yuv, hls, hsv, measure = preprocessor.get_image_from_file(side, sample)

        plt.imshow(rgb)
        plt.show()

        f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15), (ax16, ax17, ax18)) = plt.subplots(6, 3, figsize=(10, 24))

        ax1.imshow(rgb[:,:,0], cmap='gray')
        ax1.set_title("RGB R-Channel")
        ax2.imshow(rgb[:,:,1], cmap='gray')
        ax2.set_title("RGB G-Channel")
        ax3.imshow(rgb[:,:,2], cmap='gray')
        ax3.set_title("RGB B-Channel")

        ax4.imshow(yuv[:,:,0], cmap='gray')
        ax4.set_title("YUV Y-Channel")
        ax5.imshow(yuv[:,:,1], cmap='gray')
        ax5.set_title("YUV U-Channel")
        ax6.imshow(yuv[:,:,2], cmap='gray')
        ax6.set_title("YUV V-Channel")

        ax7.imshow(hls[:,:,0], cmap='gray')
        ax7.set_title("HSL H-Channel")
        ax8.imshow(hls[:,:,1], cmap='gray')
        ax8.set_title("HSL S-Channel")
        ax9.imshow(hls[:,:,2], cmap='gray')
        ax9.set_title("HSL L-Channel")

        ax10.imshow(hsv[:,:,0], cmap='gray')
        ax10.set_title("HSV H-Channel")
        ax11.imshow(hsv[:,:,1], cmap='gray')
        ax11.set_title("HSV S-Channel")
        ax12.imshow(hsv[:,:,2], cmap='gray')
        ax12.set_title("HSV V-Channel")

        # append the non augmented image and measurement
        images.append(yuv)
        measurements.append(measure)

        # augment image: flip to simulate driving the track the outher
        # way around, so curvatures get inverted --> avoid bias to one
        # side as tracks in normal direction tends to bias on left curvatures
        fl_img, fl_measure = preprocessor.flip_image(yuv, measure)

        ax13.imshow(cv2.cvtColor(fl_img, cv2.COLOR_YUV2RGB))
        ax13.set_title("Flipped image")

        images.append(fl_img)
        measurements.append(fl_measure)

        # augment image: shift the image horizontally and vertically to
        # simulate different lane positions
        shift_img, shift_measure = preprocessor.shift_image(yuv, measure)

        ax14.imshow(cv2.cvtColor(shift_img, cv2.COLOR_YUV2RGB))
        ax14.set_title("Shifted image")

        images.append(shift_img)
        measurements.append(shift_measure)

        # augment image: simulate different lighting conditions to
        # teach the network to handle different daylight and abmient
        # light conditions
        bright_img = preprocessor.adapt_brightness(hsv)

        ax15.imshow(cv2.cvtColor(bright_img, cv2.COLOR_YUV2RGB))
        ax15.set_title("Brightness adjusted")

        images.append(bright_img)
        measurements.append(measure)

        # augment image: simulate random shadows as car seems to
        # want to avoid them and is moving out of the road
        shad_img = preprocessor.adapt_shadow(hls)

        ax16.imshow(cv2.cvtColor(shad_img, cv2.COLOR_YUV2RGB))
        ax16.set_title("Shadow adjusted")

        plt.show()

        images.append(shad_img)
        measurements.append(measure)

plt.hist(measurements, bins=40, align='mid')
plt.show()
