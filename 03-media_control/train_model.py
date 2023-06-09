import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RandomFlip, RandomContrast
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from typing import Dict, List
import config as c


def main():
    annotations= parse_annotations()
    images, labels, label_names = load_images(annotations, "./data")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_test, train_labels, test_labels = preprocess_data(X_train, X_test, y_train, y_test)
    model = create_model(label_names, X_train, train_labels, X_test, test_labels)
    model = train_model(model, X_train, train_labels, X_test, test_labels)
    model.save('gesture_recognition')
    

# helper function to load and parse annotations
def parse_annotations() -> Dict:
    annotations = dict()

    for condition in c.CONDITIONS:
        with open(f'./data/_annotations/{condition}.json') as f:
            annotations[condition] = json.load(f)

    return annotations


# helper function to pre-process images (color channel conversion and resizing)
def preprocess_image(img:List) -> List:

    if c.COLOR_CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, c.SIZE)

    return img_resized


## load images and annotations
def load_images(annotations:Dict, path:str) -> List[List, List, List]:
    images = [] # stores actual image data
    labels = [] # stores labels (as integer - because this is what our network needs)
    label_names = [] # maps label ints to their actual categories so we can understand predictions later

    for condition in c.CONDITIONS:
        for filename in os.listdir(path + "/" + condition):
            # extract unique ID from file name
            UID = filename.split('.')[0]
            img = cv2.imread(f'{path}/{condition}/{filename}')
            
            # get annotation from the dict we loaded earlier
            try:
                annotation = annotations[condition][UID]
            except Exception as e:
                print(e)
                continue
            
            # iterate over all hands annotated in the image
            for i, bbox in enumerate(annotation['bboxes']):
                # annotated bounding boxes are in the range from 0 to 1
                # therefore we have to scale them to the image size
                x1 = int(bbox[0] * img.shape[1])
                y1 = int(bbox[1] * img.shape[0])
                w = int(bbox[2] * img.shape[1])
                h = int(bbox[3] * img.shape[0])
                x2 = x1 + w
                y2 = y1 + h
                
                # crop image to the bounding box and apply pre-processing
                crop = img[y1:y2, x1:x2]
                preprocessed = preprocess_image(crop)
                
                # get the annotated hand's label
                # if we have not seen this label yet, add it to the list of labels
                label = annotation['labels'][i]
                if label not in label_names:
                    label_names.append(label)
                
                label_index = label_names.index(label)
                
                images.append(preprocessed)
                labels.append(label_index)

    return images, labels, label_names 

        
def preprocess_data(X_train:List, X_test:List, y_train:List, y_test:List) -> List[List, List, List, List]:
    X_train = np.array(X_train).astype('float32')
    X_train = X_train / 255.

    X_test = np.array(X_test).astype('float32')
    X_test = X_test / 255.

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    train_label = y_train_one_hot
    test_label = y_test_one_hot

    X_train = X_train.reshape(-1, c.IMG_SIZE, c.IMG_SIZE, c.COLOR_CHANNELS)
    X_test = X_test.reshape(-1, c.IMG_SIZE, c.IMG_SIZE, c.COLOR_CHANNELS)

    return X_train, X_test, train_label, test_label


def create_model(label_names:List) -> Sequential:

    num_classes = len(label_names)
    # define model structure
    model = Sequential()

    # data augmentation (this can also be done beforehand - but don't augment the test dataset!)
    model.add(RandomFlip('horizontal'))
    model.add(RandomContrast(0.1))
    #model.add(RandomBrightness(0.1))
    #model.add(RandomRotation(0.2))

    # first, we add some convolution layers followed by max pooling
    model.add(Conv2D(64, kernel_size=(9, 9), activation=c.ACTIVATION_CONV, input_shape=(c.SIZE[0], c.SIZE[1], c.COLOR_CHANNELS), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

    model.add(Conv2D(32, (5, 5), activation=c.ACTIVATION_CONV, padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

    model.add(Conv2D(32, (3, 3), activation=c.ACTIVATION_CONV, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # dropout layers can drop part of the data during each epoch - this prevents overfitting
    model.add(Dropout(0.2))

    # after the convolution layers, we have to flatten the data so it can be fed into fully connected layers
    model.add(Flatten())

    # add some fully connected layers ("Dense")
    for i in range(c.LAYER_COUNT - 1):
        model.add(Dense(c.NUM_NEURONS, activation=c.ACTIVATION))

    model.add(Dense(c.NUM_NEURONS, activation=c.ACTIVATION))

    # for classification, the last layer has to use the softmax activation function, which gives us probabilities for each category
    model.add(Dense(num_classes, activation='softmax'))

    # specify loss function, optimizer and evaluation metrics
    # for classification, categorial crossentropy is used as a loss function
    # use the adam optimizer unless you have a good reason not to
    model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

    return model


## now, we can train the model using the fit() function
## this will take a while
def train_model(model:Sequential, X_train:List, train_label:List, X_test:List, test_label:List) -> Sequential:
    # define callback functions that react to the model's behavior during training
    # in this example, we reduce the learning rate once we get stuck and early stopping
    # to cancel the training if there are no improvements for a certain amount of epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    stop_early = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        X_train,
        train_label,
        batch_size=c.BATCH_SIZE,
        epochs=c.EPOCHS,
        verbose=1,
        validation_data=(X_test, test_label),
        callbacks=[reduce_lr, stop_early]
    )
    print(model.summary())

    return model





if __name__ == "__main__":
    main()

