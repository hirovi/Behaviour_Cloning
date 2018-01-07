import csv
import os
import cv2
import sklearn
import numpy as np
from sklearn.utils import shuffle

#Read the data file and append each row
lines = []
with open('data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Split data into training and validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

batch_size = 32

#Define a python generator which will allow reduce the data being fed into the model
def generator(samples, batch_size):
    num_samples = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, augmented_images, measurements,  augmented_measurements = [], [], [], []

            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3]) #In the CSV is a string so you need to cast it as a float
                correction = 0.2 # Correction parameter added to the steering value of the side images from the car
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                #Generalize data reading from Linux or Windows
                for i in range(3):
                    source_path = batch_sample[i]
                    if '\\' in source_path:
                        filename = source_path.split('\\')[-1]
                    else:
                        filename = source_path.split('/')[-1]

                    #Save, read and store the image
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)

                    if i == 1:
                        measurements.append(steering_left)
                    elif i == 2:
                        measurements.append(steering_right)
                    else:
                        measurements.append(steering_center)
            #Go through every image and steering angle and add the flipped image (negative steering)
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            #Convert the images and measurements into numpy arrays (Keras needs it)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

#Read next val of the generator
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

##Architecture##
#Based on the NVIDIA End to End Learning Paper for Self Driving Cars

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop 70 pixels of the top, 25 pixels of the bottom, no pixels of the left, no pixels of the right
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu')) #24 filters, 5x5 each filter, subsampes are the same as strides in keras
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu')) #36 filters, 5x5 each filter
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu')) #48 filters, 5x5 each filter
model.add(Convolution2D(64,3,3, activation='relu')) #64 filters, 3x3 each filter
model.add(Convolution2D(64,3,3, activation='relu')) #64 filters, 3x3 each filter
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Use the Adam Optimizer to minimize cost function
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
exit()
