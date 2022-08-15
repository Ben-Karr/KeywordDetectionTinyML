import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import math
import os
import random
from pprint import pprint

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

import audiomentations

def get_meta(meta_dict={}, **kwargs):
    """ 
        Retrieve the parameters from the provided dict or, if it doesn't exist, use a default value.
        If you want to pass just a few custom parameters you can use kwarg arguments otherwise you can pass 
        them as a dict. Function throws an error if the same keyword is passed more than once to make sure that the 
        intended value is used. Defaul values are from https://colab.research.google.com/github/tinyMLx/colabs/blob/master/4-6-8-CustomDatasetKWSModel.ipynb
    """
    merged_meta =  {**meta_dict, kwargs}
    assert len(merged_meta) == len(meta_dict) + len(kwargs), "It appears that a key was set more than once."
    
    training, audio, augments = {}, {}, {}
    training['wanted_words']            = merged_meta.get('wanted_words', ['on', 'off'])
    training['data_path']               = merged_meta.get('data_path', 'dataset/')
    training['epochs']                  = merged_meta.get('epochs', 5)
    training['learning_rate']           = merged_meta.get('learning_rate', 1e-3)
    training['batch_size']              = merged_meta.get('batch_size', 32)
    training['excluded_words']          = merged_meta.get('excluded_words', [])

    audio['sample_rate']                = merged_meta.get('sample_rate', 16_000)
    audio['clip_duration']              = merged_meta.get('clip_duration', 1000)
    audio['window_size_ms']             = merged_meta.get('window_size_ms', 30)
    audio['window_stride']              = merged_meta.get('window_stride', 20)
    audio['feature_bin_count']          = merged_meta.get('feature_bin_count', 40)
    audio['desired_samples']            = int(audio['sample_rate'] * audio['clip_duration'] / 1000)
    window_size_samples                 = int(audio['sample_rate'] * audio['window_size_ms'] / 1000)
    window_stride_samples               = int(audio['sample_rate'] * audio['window_stride'] / 1000)
    length_minus_window                 = audio['desired_samples'] - window_size_samples
    spectrogram_lenght                  = 1 + int(length_minus_window / window_stride_samples)
    audio['spectrogram_lenght']         = spectrogram_lenght
    audio['fingerprint_size']           = spectrogram_lenght * audio['feature_bin_count']
    
    augments['background_frequency']    = merged_meta.get('background_frequency', 0.8)
    augments['background_volume_range'] = merged_meta.get('background_volume_range', 0.1)
    augments['time_shift_ms']           = merged_meta.get('time_shift_ms', 100.0)
    augments['silence_percentage']      = merged_meta.get('silence_percentage', 0.2)
    augments['unknown_percentage']      = merged_meta.get('unknown_percentage', 0.2)
        
    return dict(training=training, audio=audio, augmentation=augments)

def get_pretrain_words(path, excluded_words, shuffle=False, n = 25):
    """
    Pulls all folders/words found at `path`. Considers only those that are not in the `excluded_words` and returns 
    `n` of those.
    """
    all_folders = [folder.split('/')[-1] for folder in glob.glob(path+'*')]
    all_included_words = [folder for folder in all_folders if folder not in (excluded_words + ['_background_noise_'])]
    if shuffle:
        random.shuffle(all_included_words)
    return all_included_words[:n]

def calc_unknown_silent_n(n, p_s, p_u):
    n_s = (p_s * (n+ (p_u * n)/(1-p_u))) / (1 - ((p_s * p_u)/(1-p_u)) - p_s)
    n_u = (p_u * (n+n_s)/(1-p_u))
    return math.ceil(round(n_s, 3)), math.ceil(round(n_u, 3))

def get_fns(path, wanted_words, excluded_words = [], val_pct = 0.2, silent_pct = 0.2, unknown_pct = 0.2, seed = None):
    """
    path:           Where to search for *.wav files
    wanted_words:   The keywords that should be detected
    excluded_wods:  Words that should not be used in training. 
                    Words that are neither wanted nor excluded will be used as unknown_words
    val_pct:        Percentage of files that should be used for validation
    unknown_pct:    Percentage of the train/val split that are unknown
    silent_pct:     Percentage of the train/val split that are labeled silent
    """
    wanted_fns_dict = {}
    unknown_fns = []
    background_fns = []
    ## Get all but excluded .wav files contained at the provided path and add them to the appropriate list
    wavs = glob.glob(os.path.join(path,'*','*.wav'))
    for fn in wavs:
        folder = os.path.split(os.path.dirname(fn))[-1]
        if folder in excluded_words:
            continue
        if folder == '_background_noise_':
            background_fns.append(fn)
        elif folder in wanted_words:
            ## Creates a list containing fn at keyword if the keyword isn't contained in the dict yet,
            ## else adds fn to list.
            if wanted_fns_dict.get(folder, False):
                wanted_fns_dict[folder].append(fn)
            else:
                wanted_fns_dict[folder] = [fn]
        else:
            unknown_fns.append(fn)
            
    ## Split wanted/unknown in training and validation, for each wanted words: val_pct of the total number of 
    ## per word fns are in the validation set (1-val_pct) in the training set.
    training_words = []
    validation_words = []
    for key in wanted_fns_dict.keys():
        word_fns = wanted_fns_dict[key]
        random.shuffle(word_fns)
        n_val_word = int(len(word_fns) * val_pct)
        validation_words.extend(word_fns[:n_val_word])
        training_words.extend(word_fns[n_val_word:])
    
    ## Calcs number of silent/unknown for train/val split to hit certain percentage
    n_silent_train, n_unknown_train = calc_unknown_silent_n(len(training_words), silent_pct, unknown_pct)
    n_silent_val, n_unknown_val = calc_unknown_silent_n(len(validation_words), silent_pct, unknown_pct)
    
    ## Keep validation determenistic
    validation_unknown = unknown_fns[:n_unknown_train]
    non_validation_unknown = unknown_fns[n_unknown_train:]
    ## Pick training unknowns at random
    random.shuffle(non_validation_unknown)
    training_unknown = non_validation_unknown[:n_unknown_train]
    
    training_fns = training_words + training_unknown + ['silence_placeholder'] * n_silent_train
    validation_fns = validation_words + validation_unknown + ['silence_placeholder'] * n_silent_val
        
    random.shuffle(training_fns)
    random.shuffle(validation_fns)
    random.shuffle(background_fns)
    
    return training_fns, validation_fns, background_fns

class KeywordDataset(tf.keras.utils.Sequence):
    def __init__(self,
                 fns,
                 background_fns,
                 meta_dict,
                 batch_size,
                 is_validation = False
                ):
        self.items = fns
        self.words = meta_dict['training']['wanted_words']
        self.vocab = {word: i for i,word in enumerate(['silence', 'unknown'] + self.words)}
        self.audio_meta = meta_dict['audio']
        self.augmentation_meta = meta_dict['augmentation']
        self.is_validation = is_validation
        self.augment = self.build_augments()
        self.background_data = self.prepare_background_data(background_fns)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.items) / self.batch_size)

    def __getitem__(self, idx):
        """
        Pulls a subset of filenames of size `batch_size`. Loads the audio file according to its label and adds 
        augmentations if in 'training mode'. Finally creates a spectrogram and combines the batch to single 
        numpy x,y vectors.
        """
        items = self.items[idx * self.batch_size: (idx + 1) * self.batch_size]
        xs, ys = [], []
        for fn in items:
            label = self.get_label(fn)
            audio = self.get_audio(fn, label).numpy().flatten()
            if not self.is_validation:
                audio = self.augment(audio, sample_rate = self.audio_meta['sample_rate'])
            spectro = self.get_spectrogram(audio)       
            xs.append(spectro)
            ys.append(self.vocab[label])
        return np.stack(xs), np.stack(ys)
    
    def on_epoch_end(self):
        if not self.is_validation:
            random.shuffle(self.items)

    def prepare_background_data(self,fns):
        ## See `prepare_background_data` in tensorflow/examples/speech_commands/input_data.py
        background_data = []
        for fn in fns:
            file = tf.io.read_file(fn)
            audio, _ = tf.audio.decode_wav(file, desired_channels=1)
            if len(audio) < self.audio_meta['desired_samples']:
                continue
            background_data.append(audio)
        return background_data
        
    def get_label(self, fn):
        if fn == 'silence_placeholder':
            return 'silence'
        else:
            folder = fn.split('/')[-2]
            if folder in self.words:
                return folder
            return 'unknown'
    
    def load_audio(self, fn):
        file = tf.io.read_file(fn)
        audio, _ = tf.audio.decode_wav(contents = file, 
                                       desired_channels = 1, 
                                       desired_samples = self.audio_meta['desired_samples']
                                      )     
        return audio
    
    def get_timeshift_params(self):
        ## See `get_data` in tensorflow/examples/speech_commands/input_data.py
        time_shift = self.augmentation_meta['time_shift_ms']
        background_frequency = self.augmentation_meta['background_frequency']
        background_volume_range = self.augmentation_meta['background_volume_range']
        
        time_shift_amount = np.random.randint(-time_shift, time_shift) if time_shift > 0 else 0
        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0,0]]
            time_shift_offset = [0,0]
        else:
            time_shift_padding = [[0,-time_shift_amount], [0,0]]
            time_shift_offset = [-time_shift_amount, 0]
            
        return time_shift_padding, time_shift_offset
    
    def get_random_background(self, label):
        ## See `get_data` in tensorflow/examples/speech_commands/input_data.py
        background_sample = random.choice(self.background_data)

        background_offset = np.random.randint(0, len(background_sample) - self.audio_meta['desired_samples'])
        background_clipped = background_sample[background_offset:(background_offset + self.audio_meta['desired_samples'])]
        background_reshaped = tf.reshape(background_clipped, [self.audio_meta['desired_samples'],1])
        
        if label == 'silence':
            background_volume = np.random.uniform(0,1)
        elif np.random.uniform(0,1) < self.augmentation_meta['background_frequency']:
            background_volume = np.random.uniform(0, self.augmentation_meta['background_volume_range'])
        else:
            background_volume = 0

        background_mul = tf.multiply(background_reshaped, background_volume)
        return background_mul
    
    def get_audio(self, fn, label):
        """
        Adds random background to audio and shifts it a bit back or forth if in 'training mode', 
        returns background-only if label is `silence`.
        """
        if self.is_validation and label != 'silence':
            return self.load_audio(fn)
        background_mul = self.get_random_background(label)
        if label == 'silence':
            return background_mul
        
        ## See `prepare_processing_graph` in tensorflow/examples/speech_commands/input_data.py
        foreground = self.load_audio(fn)
        time_shift_padding, time_shift_offset = self.get_timeshift_params()
        
        padded_foreground = tf.pad(tensor = foreground, paddings = time_shift_padding, mode = 'CONSTANT')
        sliced_foreground = tf.slice(padded_foreground, time_shift_offset, [self.audio_meta['desired_samples'], -1])
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1., 1.)
        
        return background_clamp
        
    def get_spectrogram(self, audio):
        ## See `prepare_processing_graph` in tensorflow/examples/speech_commands/input_data.py
        int_16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int_16_input,
            sample_rate = self.audio_meta['sample_rate'],
            window_size = self.audio_meta['window_size_ms'],
            window_step = self.audio_meta['window_stride'],
            num_channels = self.audio_meta['feature_bin_count'],
            out_scale = 1,
            out_type = tf.float32
        )
        flat_spectro = tf.multiply(micro_frontend, (10. / 256.)).numpy().flatten()
        return flat_spectro
    
    def build_augments(self):
        ## Uses the audiomentations library. Check https://github.com/iver56/audiomentations for further details.
        augs = audiomentations.Compose([
            audiomentations.ClippingDistortion(max_percentile_threshold=20, p=.5),
            audiomentations.HighPassFilter(min_cutoff_freq=1000, p=.3),
            audiomentations.LowPassFilter(min_cutoff_freq=1000, p=.3),
            audiomentations.GainTransition(min_gain_in_db=-12,max_gain_in_db=12,min_duration=0.1,max_duration=0.9,duration_unit='fraction',p=.5),
            audiomentations.PitchShift(min_semitones=-1, max_semitones=1, p=.3),
            audiomentations.SevenBandParametricEQ(p=.5),
            audiomentations.PolarityInversion(p=0.3),
            audiomentations.TimeMask(p=.3),
            audiomentations.AddGaussianNoise(max_amplitude=0.005, p=0.3),
        ])
        return augs