import numpy as np
import librosa
import random
import pandas as pd

def feature_extraction(extracted_wav):
    features = []
    for item in extracted_wav:
        sr = item["sr"] 
        y = item["y"]

        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spctrl_ctrd = librosa.feature.spectral_centroid(y=y, sr=sr)
        spctrl_bdwth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spctrl_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)

        features.append(np.hstack([
            np.mean(y), np.std(y), np.max(y), np.min(y), # Statistik dasar
            np.mean(zcr), np.mean(rms),                  # Fitur waktu
            np.mean(spctrl_ctrd), np.std(spctrl_ctrd),   # Spectral Centroid
            np.mean(spctrl_bdwth), np.std(spctrl_bdwth), # Spectral Bandwidth
            np.mean(spctrl_rolloff), np.std(spctrl_rolloff), # Rolloff
            np.mean(mfcc_feat, axis=1),                  # Mean MFCC (13 fitur)
            np.std(mfcc_feat, axis=1),                   # Std MFCC (13 fitur)
            item["speaker"], item["digit"], item["sample_number"],                 
        ]))
    
    return pd.DataFrame(features)

class Augmentation():
    def __init__(self, noise_factor, time_shift_max, pitch_step, t_stretch_rate, volume_scale_factor):
        self.noise_factor = noise_factor
        self.time_shift_max = time_shift_max
        self.pitch_step = pitch_step
        self.t_stretch_rate = t_stretch_rate
        self.volume_scale_factor = volume_scale_factor
    
    def noise_inject(self, audio_signal):
        noise = np.random.randn(len(audio_signal))
        augmented = audio_signal + self.noise_factor * noise
        return augmented
    
    def time_shift(self, audio_signal):
        shift = int(np.random.uniform(-self.time_shift_max, self.time_shift_max) * len(audio_signal))
        augmented = np.roll(audio_signal, shift)
        return augmented
    
    def pitch_shift(self, audio_signal, sr):
        augmented = librosa.effects.pitch_shift(y=audio_signal, sr=sr, n_steps=random.choice(self.pitch_step))
        return augmented
    
    def time_stretch(self, audio_signal):
        augmented = librosa.effects.time_stretch(y=audio_signal, rate=self.t_stretch_rate)
        return augmented
    
    def volume_scale(self, audio_signal):
        augmented = audio_signal * random.choice(self.volume_scale_factor)
        return augmented

# def feature_extraction(audio_signal):
#     y, sr = librosa.load(audio_signal)
#     y, _ = librosa.effects.trim(y)

#     mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     spctrl_ctrd = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spctrl_bdwth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     spctrl_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     rms = librosa.feature.rms(y=y)
#     zcr = librosa.feature.zero_crossing_rate(y=y)

#     features = np.hstack([
#         np.mean(y), np.std(y), np.max(y), np.min(y), # Statistik dasar
#         np.mean(zcr), np.mean(rms),                  # Fitur waktu
#         np.mean(spctrl_ctrd), np.std(spctrl_ctrd),   # Spectral Centroid
#         np.mean(spctrl_bdwth), np.std(spctrl_bdwth), # Spectral Bandwidth
#         np.mean(spctrl_rolloff), np.std(spctrl_rolloff), # Rolloff
#         np.mean(mfcc_feat, axis=1),                  # Mean MFCC (13 fitur)
#         np.std(mfcc_feat, axis=1)                    # Std MFCC (13 fitur)
#     ])
    
#     return features