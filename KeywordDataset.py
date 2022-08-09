import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import math
import os
import random

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

class KeywordDataset(tf.keras.utils.Sequence):
    def __init__(self,
                 fns,
                 background_fns,
                 meta_dict,
                 batch_size,
                 is_validation = False
                ):
        self.batch_size = batch_size
        self.words = meta_dict['training']['wanted_words']
        self.vocab = {word: i for i,word in enumerate(['silence', 'unknown'] + self.words)}
        self.unknown_fns = fns[1]
        self.audio_meta = meta_dict['audio']
        self.augmentation_meta = meta_dict['augmentation']
        self.is_validation = is_validation

        self.items = self.prepare_items(fns[0])
        self.background_data = self.prepare_background_data(background_fns)

    def __len__(self):
        return math.ceil(len(self.items) / self.batch_size)

    def __getitem__(self, idx):
        items = self.items[idx * self.batch_size: (idx + 1) * self.batch_size]
        xs, ys = [], []
        for fn in items:
            label = self.get_label(fn)
            audio = self.get_audio(fn, label)
            spectro = self.get_spectrogram(audio)       
            xs.append(spectro)
            ys.append(self.vocab[label])
        return np.stack(xs), np.stack(ys)
    
    def on_epoch_end(self):
        if not self.is_validation:
            random.shuffle(self.items)
        
    def prepare_items(self, items):
        """ 
        Add the same amoung of placeholders for silence as there are unknowns.
        Return a shuffled list of items.
        To-Do: move this to the filename retrieval
        """
        items = items + ['silence_placeholder'] * len(self.unknown_fns) + self.unknown_fns
        random.shuffle(items)
        return items

    def prepare_background_data(self,fns):
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
        audio, _ = tf.audio.decode_wav(contents = file, desired_channels = 1, desired_samples = self.audio_meta['desired_samples'])     
        return audio
    
    def get_timeshift_params(self):
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
        if self.is_validation and label != 'silence':
            return self.load_audio(fn)
        background_mul = self.get_random_background(label)
        if label == 'silence':
            return background_mul

        foreground = self.load_audio(fn)
        time_shift_padding, time_shift_offset = self.get_timeshift_params()
        
        padded_foreground = tf.pad(tensor = foreground, paddings = time_shift_padding, mode = 'CONSTANT')
        sliced_foreground = tf.slice(padded_foreground, time_shift_offset, [self.audio_meta['desired_samples'], -1])
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1., 1.)
        
        return background_clamp
        
    def get_spectrogram(self, audio):
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
        spectro = tf.multiply(micro_frontend, (10. / 256.)).numpy().flatten()
        return spectro

def get_fns(path, wanted_words, val_pct = 0.2, unknown_pct = 0.2, seed = None):
    wanted_words_fns = {}
    unknown_words_fns = []
    background_fns = []
    
    """ Get all .wav files contained at the provided path and add them to the appropriate list """
    fns = glob.glob(os.path.join(path,'*','*.wav'))
    for fn in fns:
        folder = os.path.split(os.path.dirname(fn))[-1]
        if folder == '_background_noise_':
            background_fns.append(fn)
        elif folder in wanted_words:
                if wanted_words_fns.get(folder, False):
                    wanted_words_fns[folder].append(fn)
                else:
                    wanted_words_fns[folder] = [fn]
        else:
            unknown_words_fns.append(fn)
            
    """ Split wanted/unknown in training and validation """
    training_words = []
    validation_words = []
    for key in wanted_words_fns.keys():
        word_fns = wanted_words_fns[key]
        random.shuffle(word_fns)
        n_val_word = int(len(word_fns) * val_pct)
        validation_words.extend(word_fns[:n_val_word])
        training_words.extend(word_fns[n_val_word:])
    
    n_val_unknown = int(len(unknown_words_fns) * val_pct)
    validation_unknowns = unknown_words_fns[:n_val_unknown]
    training_unknowns = unknown_words_fns[n_val_unknown:]
    
    n_training = len(training_words)
    n_training_unknown = int(n_training * unknown_pct)
    training_unknowns = random.sample(training_unknowns, k = n_training_unknown)
    
    n_validation = len(validation_words)
    n_validation_unknown = int(n_validation * unknown_pct)
    validation_unknowns = random.sample(validation_unknowns, k = n_validation_unknown)
    
    training_fns = [training_words, training_unknowns]
    validation_fns = [validation_words, validation_unknowns]
    return training_fns, validation_fns, background_fns