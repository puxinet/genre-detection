import os
import h5py
import librosa
import audio_processor as ap
import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import floor

# Loads a dataset of Mel-Spectrum
def load_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('labels'))
        num_frames = np.array(hf.get('num_frames'))
    return data, labels, num_frames

# Saves a dataset of Mel-Spectrum
def save_dataset(path, data, labels, num_frames):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('num_frames', data=num_frames)

# Melgram computation
def extract_melgrams(list_path, process_all_song, num_songs_genre):
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    song_paths = open(list_path, 'r').read().splitlines()
    labels = list()
    num_frames_total = list()
    for song_ind, song_path in enumerate(song_paths):
        melgram = ap.compute_melgram(song_path)
        index = int(floor(song_ind/num_songs_genre))
        labels.append(index)
        melgrams = np.concatenate((melgrams, melgram), axis=0)
    return melgrams, labels, num_frames_total
