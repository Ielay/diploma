from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU

import numpy as np
import csv
import sys

##############
picture_h = 45
picture_w = 45

# dictionary of labels and their number for one-hot encoding
signs_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,

    "a": 10,
    "b": 11,
    "c": 12,
    "d": 13,
    "e": 14,
    "f": 15,
    "g": 16,
    "h": 17,
    "i": 18,
    "j": 19,
    "k": 20,
    "l": 21,
    "m": 22,
    "n": 23,
    "o": 24,
    "p": 25,
    "q": 26,
    "r": 27,
    "s": 28,
    "t": 29,
    "u": 30,
    "v": 31,
    "w": 32,
    "x": 33,
    "y": 34,
    "z": 35,

    "less": 36,
    "less_or_eq": 37,
    "equal": 38,
    "greater": 39,
    "greater_or_eq": 40,
    "left_bracket": 41,
    "right_bracket": 42,
    "plus": 43,
    "minus": 44,
    "times": 45,
    "div": 46
}

# reading training and testing data
train_label = []
train_data = []

filename = sys.argv[1]

with open(filename) as csv_train_data:
    csv_reader = csv.reader(csv_train_data)
    for row in csv_reader:
        # set label (10, 1)
        # new_label = [0 for i in range(9)]
        # new_label.insert(int(row[0]), 1)
        #print(row)
        #print("Row[0]:", row[0])
        #print("int(row[0]):", int(row[0]))
        new_label = [0 for i in range(len(signs_dict) - 1)]
        new_label.insert(signs_dict[row[0]], 1)
        train_label.append(new_label)
        # set data (784, 1)
        train_data.append(row[1:len(row)])

train_data = np.array([np.reshape(np.array(vect), (picture_w, picture_h, 1)) for vect in train_data])

#test_label = []
#test_data = []

#with open("mnist_test.csv") as csv_test_data:
   # csv_reader = csv.reader(csv_test_data)
   # for row in csv_reader:
      #  # set label (10, 1)
     #   new_label = [0 for i in range(9)]
    #    new_label.insert(int(row[0]), 1)
   #     test_label.append(new_label)
  #      # set data (784, 1)
 #       test_data.append(row[1:len(row)])

#test_data = np.array([np.reshape(np.array(vect), (picture_w, picture_h, 1)) for vect in test_data])
##################
model = Sequential()

# 1
model.add(Conv2D(32, kernel_size=(7, 7), strides=(2, 2),
          activation='sigmoid', input_shape=(picture_w, picture_h, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 2
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# 3
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
          activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten()) # input

model.add(Dense(2000, activation='tanh')) # hidden 1
model.add(Dense(1500)) # hidden 2
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1000)) # hidden 3
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(len(signs_dict), activation='sigmoid')) # output

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# training
model.fit(np.array(train_data), np.array(train_label), epochs=20, batch_size=32)

# test
#score = model.evaluate(np.array(test_data), np.array(test_label), batch_size=32)
#print(score)

# save
model.save("my_nn_convolution_model.h5")
