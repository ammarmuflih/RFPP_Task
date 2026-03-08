import numpy as np
import librosa
import random

def feature_extraction(audio_signal):
    y, sr = librosa.load(audio_signal)
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spctrl_ctrd = librosa.feature.spectral_centroid(y=y, sr=sr)
    spctrl_bdwth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spctrl_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    feature = {
        'mean': np.mean(y),
        'std': np.std(y),
        'max': np.max(y),
        'min': np.min(y),
        'median': np.median(y),
        'rms': np.sqrt(np.mean(y**2)),
        'zcr': np.mean(np.abs(np.diff(np.sign(y)))),
       
        'centroid_mean': np.mean(spctrl_ctrd),
        'centroid_std': np.std(spctrl_ctrd),
        'centroid_min': np.min(spctrl_ctrd),
        'centroid_max': np.max(spctrl_ctrd),

        'bandwidth_mean': np.mean(spctrl_bdwth),
        'bandwidth_std': np.std(spctrl_bdwth),
        'bandwidth_min': np.min(spctrl_bdwth),
        'bandwidth_max': np.max(spctrl_bdwth),

        'rolloff_mean': np.mean(spctrl_rolloff),
        'rolloff_std': np.std(spctrl_rolloff),
        'rolloff_min': np.min(spctrl_rolloff),
        'rolloff_max': np.max(spctrl_rolloff),

        'mean_mfcc': np.mean(mfcc_feat, axis=1),
        'std_mfcc': np.std(mfcc_feat, axis=1)
    }
    return feature

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