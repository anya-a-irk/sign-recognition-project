import numpy as np
from skimage import color, exposure, transform, io
import os
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import pandas as pd
import h5py

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in v channel (last dimension of HSV format)
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # Central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # Rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

    # Roll color axis (axis -1, last axis) to axis 0
    # img = np.rollaxis(img,-1)  # no need since we use tf format.

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
              input_shape=(IMG_SIZE, IMG_SIZE, 3)))
        # nb_filters (Keras 1) --> filters (Keras 2). Number of filters = 32. 
        # kernel_size[0] = 3, kernel_size[1] = 3.
        # border_mode (Keras 1) --> padding (Keras 2)
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def train():
    root_dir = os.path.expanduser('/home/anna/project/GTSRB/Final_Training/Images/')
        # expanduser() lets python know the meaning of "~" 
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))  
        # glob() returns all matching file path
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    model = cnn_model()

    # Train the model using SGD + momentum
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    def lr_schedule(epoch):
        return lr * (0.1 ** int(epoch / 10))
    print(X.shape)
    print(Y.shape)
    batch_size = 32
    epochs = 20
    model.fit(X, Y,
              batch_size=batch_size,
              epochs = epochs,
              validation_split=0.2,
              callbacks=[LearningRateScheduler(lr_schedule), 
                         ModelCheckpoint('model.h5', save_best_only=True)])

def test():
    model = cnn_model()
    model.load_weights('model.h5')

    # Evaluation
    test = pd.read_csv(os.path.expanduser('/home/anna/project/GT-final_test.csv'), sep=';')

    # Load test dataset
    X_test = []
    Y_test = []
    i = 0
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('/home/anna/project/GTSRB/Final_Test/Images/', file_name)
        img_path = os.path.expanduser(img_path)
        X_test.append(preprocess_img(io.imread(img_path)))
        Y_test.append(class_id)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Predict and evaluate
    Y_pred = model.predict_classes(X_test)
    acc = np.sum(Y_pred == Y_test) / np.size(Y_pred)
    print("Test accuracy = {}".format(acc))

def main():
    train()
    test()

if __name__ == '__main__':
    main()