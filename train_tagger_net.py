import os
import sys
from tagger_net import MusicTaggerCNN, MusicTaggerCRNN
import numpy as np
from keras.utils import np_utils
from utils import load_dataset, save_dataset, extract_melgrams
from keras.utils.vis_utils import plot_model

# Configuration parameters
SAVE = 0 # Save the Mel-Spectrum
LOAD = 1 # Load the Mel-Spectrum
CRNN = 1 # Use CRNN


# GTZAN Dataset 
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)
train_songs_list = 'lists/train_songs_list.txt'
test_songs_list = 'lists/test_songs_list.txt'

# Indicate the name of train and test songs
train_gt_list = 'lists/train_gt_list.txt'
test_gt_list = 'lists/test_gt_list.txt'

# Data Loading or computing the Mel-spectogram for each song
if LOAD:
    X_train, y_train, num_frames_train = load_dataset('music_dataset/music_dataset_train.h5')
    X_test, y_test, num_frames_test = load_dataset('music_dataset/music_dataset_test.h5')
else:
    print('Computing melgrams for training dataset')
    X_train, y_train, num_frames_train = extract_melgrams(train_songs_list, process_all_song=False, num_songs_genre=70)
    print('X_train shape:', X_train.shape)
    print('Computing melgrams for testing dataset')
    X_test, y_test, num_frames_test = extract_melgrams(test_songs_list, process_all_song=False, num_songs_genre=30)
    print('X_train shape:', X_train.shape)

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')


y_train = np.array(y_train)
y_test = np.array(y_test)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print('Shape labels y_train: ', Y_train.shape)
print('Shape labels y_test: ', Y_test.shape)

# Save database for reuse
if SAVE:
    save_dataset('music_dataset/music_dataset_train.h5', X_train, y_train,num_frames_train)
    save_dataset('music_dataset/music_dataset_test.h5', X_test,y_test,num_frames_test)


# Initialize model
if CRNN:
    model = MusicTaggerCRNN()
else:
    model = MusicTaggerCNN()

#Plot model information
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#Train model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50)