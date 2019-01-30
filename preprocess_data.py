"""
Author - Torstein Gombos
Date - 08.01.2019

This script reads CSV files and uses JSON to to navigate the data that exists on YouTube.
"""

import json
import pandas as pd
import wave
import librosa
import random
import numpy
import csv
import re


def fetch_labels_youtube(file_path="../Datasets/balanced_train_segments.csv"):
    """
    Load the csv file for the YouTube data set
    :return:
    """

    with open(file_path, 'r') as csvfile:
        df = csv.reader(csvfile, delimiter=' ', quotechar='"')
        i = 0
        for row in df:
            i += 1
            print(', '.join(row))
        print(f'There is {i} samples in the data')
        print(df)


def plot_time_signal():
    """
    Plot the time signal of the audio files
    :return:
    """


def load_wav_files(filepath='../Datasets/UrbanSound8K/audio/fold1'):
    """
    Use Librosa to load the wav files from a specific fodler
    :param filepath:
    :return:
    """
    librosa.load(filepath, sr=22050)


def fetch_labels():
    """
    Creates a dict of classes for the labels in the UrbanSounds dataset
    :return:
    """
    label_dict = {}
    list_of_classes = ['air_conditioner',
                       'car_horn',
                       'children_playing',
                       'dog_bark',
                       'drilling',
                       'engine_idling',
                       'gunshot',
                       'jackhammer',
                       'siren',
                       'street_music']
    for i, c in enumerate(list_of_classes):
        label_dict[i] = c
    return label_dict



if __name__ == '__main__':
    """

    :return:
    """
    # Create a dict with the labels that match the idx number for the readme file for UrbanSounds
    labels = fetch_labels()
    print(labels)



