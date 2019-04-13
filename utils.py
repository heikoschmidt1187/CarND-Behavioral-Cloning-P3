import csv
import matplotlib.pyplot as plt
import numpy as np

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

angles = []

for sample in samples:
    angles.append(float(sample[3]))
    angles.append(float(sample[3]) + 0.2)
    angles.append(float(sample[3]) - 0.2)
    angles.append(-1.0 * float(sample[3]))
    angles.append(-1.0 * (float(sample[3]) + 0.2))
    angles.append(-1.0 * (float(sample[3]) - 0.2))


    # shift images for lane switches
    for i in range(3):
        x_translation = 80 * np.random.uniform() - 80/2

        if i == 0:
            steering = float(sample[3])
        elif i == 1:
            steering = float(sample[3]) + 0.2
        else:
            steering = float(sample[3]) - 0.2

        steering_ang = steering + x_translation / 20*2*0.2
        angles.append(steering_ang)


plt.hist(angles, bins=40, align='mid')
plt.show()
