"""
Script to identify whale calls using spectogram.
"""

import argparse
import boto3
import pandas as pd
import os
import json
from os import path
from os.path import expanduser
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.models import model_from_json
import glob
from sklearn.metrics import accuracy_score, roc_auc_score


class whaledr(object):
    """
        Class to access spectogram data from s3 and Pre-train classifier.
    """
    def __init__(self):
        self.CREDS_DATA = {}
        self.BUCKET_NAME = 'whaledr'
        self.WHALE_PATH = '../model/data/train_images/whale'
        self.NON_WHALE_PATH = '../data/train_images/non_whale'
        self.PREFIX = 'megaptera'
        self.LOOKUP_DELIMITER = '___'
        self.START_LEN = len(self.PREFIX.split('/')[0]) + 1
        self.WHALE_LOOKUP_FILE = '../model/data/megaptera-13ae9-sampleSummary-export.json'
        self.HYDROPHONE_NAME = ('LJ01C', 'LJ01A') # make this a tuple ('LJ01C', 'LJ01A')
        self.TRAIN_DIR = os.path.join(os.getcwd() + os.sep + 'train_images')
        self.CATEGORIES = ['whale', 'non_whale']
        self.INPUT_SIZE = 299
        self.TEST_SIZE_SPLIT = 0.1
        self.SEED = 29
        self.BATCH_SIZE = 16
        self.EPOCS = 10
        self.LAST_LAYERS = 10
        self.MODEL_WEIGHT_CHECKPOINT_PATH = '../model/data/model_weights'

    def load_creds(self):
        """
            Utility function to read s3 credential file for
            data upload to s3 bucket.
        """
        home = expanduser("~")
        with open(os.path.join(home, 'creds.json')) as creds_file:
            self.CREDS_DATA = json.load(creds_file)

    def data_prep(self):
        """
            Utility function to read firebase app lookup file 
            and pre-process whale and non-whale metadata.
        """
        # read whale-dr json file to get labelled images
        with open(self.WHALE_LOOKUP_FILE) as whale_file:
            whale_lookup = json.load(whale_file)
        # get all keys from the json
        all_val = list(whale_lookup.keys())
        # get image id from resulting list checking keys starting with hydrophone_name_pattern.
        image_id = list(filter(lambda x: x.startswith(self.HYDROPHONE_NAME), all_val))
        final_data = pd.DataFrame(columns=['image_id', 'num_votes', 'avg_score'])
        num_votes = []
        avg_score = []
        for idx, id in enumerate(image_id):
            num_votes.append(whale_lookup[id]['count'])
            avg_score.append(whale_lookup[id]['aveVote'])
        final_data['image_id'], final_data['num_votes'], final_data['avg_score'] = image_id, num_votes, avg_score
        # data frame with image_id and score
        final_data['image_id'] = self.PREFIX + self.LOOKUP_DELIMITER + final_data['image_id'] + '.jpg'
        final_data = final_data[final_data['num_votes'] != 0]
        # get whale and non-whale images based on atleast 1 vote and avg score > 0 being whales.
        self.whale_images = list(final_data[(final_data['num_votes'] != 0) & (final_data['avg_score'] > 0)]['image_id'])
        self.non_whale_images = list(final_data[(final_data['num_votes'] != 0) & (final_data['avg_score'] == 0)]['image_id'])

        # adding labels to the data frame
        final_data['category'] = np.where(final_data['avg_score'] > 0, 'whale', 'non_whale')
        final_data['image_id'] = np.where(final_data['avg_score'] > 0, 'whale' + os.sep + final_data['image_id'], 'non_whale' + os.sep + final_data['image_id'])
        self.final_data = final_data


    def data_fetch(self):
        """
            Utility function to read spectogram data saved on
            s3 bucket and create necessary folder struture of model
            training.
        """
        # Access Spectogram from S3
        if not os.path.exists(self.WHALE_PATH):
            os.makedirs(self.WHALE_PATH)
        if not os.path.exists(self.NON_WHALE_PATH):
            os.makedirs(self.NON_WHALE_PATH)
        s3 = boto3.resource('s3', aws_access_key_id=self.CREDS_DATA['key_id'],
                 aws_secret_access_key=self.CREDS_DATA['key_access'])
        bucket = s3.Bucket(self.BUCKET_NAME)
        home = os.getcwd()
        for obj in bucket.objects.filter(Delimiter='', Prefix=self.PREFIX):
            if self.LOOKUP_DELIMITER.join(obj.key.split('/')) in self.whale_images:
                bucket.download_file(obj.key, os.path.join(os.path.join(home, self.WHALE_PATH, self.LOOKUP_DELIMITER.join(obj.key.split('/')))))
            elif self.LOOKUP_DELIMITER.join(obj.key.split('/')) in self.non_whale_images:
                bucket.download_file(obj.key, os.path.join(os.path.join(home, self.NON_WHALE_PATH, self.LOOKUP_DELIMITER.join(obj.key.split('/')))))
                
    def read_img(self, filepath):
        """
            Utility function to read spectogram image.
        """
        img = image.load_img(os.path.join(self.TRAIN_DIR, filepath), target_size=(self.INPUT_SIZE, self.INPUT_SIZE))
        img = image.img_to_array(img)
        return img

    def pre_process_input(self):
        """
            Utility function to pre-process image file for model training.
        """
        skipped_ag = []
        self.X_Train = np.zeros((len(self.final_data), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype='float32')
        for i, file_path in (enumerate(self.final_data['image_id'])):
            try:
                img = self.read_img(file_path)
                x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
                self.X_Train[i] = x
            except Exception as e:
                skipped_ag.append(i)
                pass

        # deleting images_id for corresponding missing images on s3
        self.X_Train = np.delete(self.X_Train, skipped_ag, axis=0)
        self.final_data = self.final_data.drop(self.final_data.index[skipped_ag])
        self.final_data = self.final_data.reset_index(drop=True)


    def validation_setup(self):
        """
            Utility function to create train validation setup.
        """
        # validation_split
        self.Y_train = self.final_data['category'].values
        self.Y_train = list(map(lambda x: 0 if x == 'non_whale' else 1, self.Y_train))
        self.train_x, self.train_valid, self.y_train, self.y_valid = train_test_split(self.X_Train,
                                                                    self.Y_train, test_size=self.TEST_SIZE_SPLIT, 
                                                                    random_state=self.SEED, stratify=self.Y_train)

    def model_train(self):
        """
            Utility function to use pre-train model with data augmentation
            and train last n layers.
        """
        # Base model with Transfer Learning
        baseModel = xception.Xception(weights="imagenet", include_top=False, input_shape=(self.INPUT_SIZE, self.INPUT_SIZE, 3))
        # unFreeze last n layers
        for layer in baseModel.layers[-self.LAST_LAYERS:]:
            layer.trainable = True

        # Images generate on the run so we don't have to save any of them.
        datagen = ImageDataGenerator(
                featurewise_center=False,
                # featurewise_std_normalization = True,
                shear_range=0.2,
                # zoom_range=0.2,
                rotation_range=60,
                horizontal_flip=True,
                vertical_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2)

        datagen.fit(self.train_x)

        transferModel = Sequential([
            baseModel,
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.25),
            Dense(32, activation='relu'),
            Dropout(0.20),
            Dense(1, activation='sigmoid')
        ])

        self.optimizer = Adam(0.0001, decay=0.00001, amsgrad=True)
        if not os.path.exists(self.MODEL_WEIGHT_CHECKPOINT_PATH):
            os.mkdir(self.MODEL_WEIGHT_CHECKPOINT_PATH)
        self.weightpath = self.MODEL_WEIGHT_CHECKPOINT_PATH + '/Epoch-{epoch:02d}_Val_accuracy-{val_acc:.2f}.h5'
        checkpoint = ModelCheckpoint(self.weightpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        callbacks_list = [checkpoint, es]
        transferModel.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.history = transferModel.fit_generator(datagen.flow(self.train_x, self.y_train, batch_size=self.BATCH_SIZE),
                                steps_per_epoch=len(self.train_x),
                                epochs=self.EPOCS,
                                validation_data=(self.train_valid, self.y_valid), callbacks=callbacks_list, verbose=1)
        model_json = transferModel.to_json()
        with open("{}/model.json".format(self.MODEL_WEIGHT_CHECKPOINT_PATH), "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

    def model_predict(self):
        """
            Utility function to use trained model weights and make predictions.
        """
        # load json and create model
        json_file = open("{}/model.json".format(self.MODEL_WEIGHT_CHECKPOINT_PATH), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        weight_file = glob.glob('{}/*.h5'.format(self.MODEL_WEIGHT_CHECKPOINT_PATH))[0]
        loaded_model.load_weights(weight_file)
        print("Loaded model from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        print("Accuracy is:", accuracy_score(self.y_valid, loaded_model.predict(self.train_valid).argmax(axis=-1)))
        print("ROC is:", roc_auc_score(self.y_valid, loaded_model.predict_proba(self.train_valid)))

    def main(self):
        """
            Utility function to save loss history as csv file.
        """
        loss_history = self.history.history
        epochs = range(1, len(loss_history['val_loss'])+1)
        final_data = (pd.DataFrame(loss_history, index=epochs))
        final_data.to_csv('results.csv', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train whale sound classification model')
    parser.add_argument('-f', '--flag', choices=['fetch_data', 'skip_data_fetch'], default="skip_data_fetch", help='wheather to get s3 data')
    args = parser.parse_args()
    whaledr = whaledr()
    whaledr.load_creds()
    whaledr.data_prep()
    if args.flag == 'fetch_data':
        whaledr.data_fetch()
        quit()
    whaledr.pre_process_input()
    whaledr.validation_setup()
    whaledr.model_train()
    whaledr.model_predict()
    whaledr.main()
