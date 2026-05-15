import numpy as np
import librosa
import random
import pandas as pd

# def feature_extraction(extracted_wav):
#     features = []
#     for item in extracted_wav:
#         sr = item["sr"] 
#         y = item["y"]

#         mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_mels=40)
#         # delta = librosa.feature.delta(mfcc_feat, width=3)
#         # delta2 = librosa.feature.delta(mfcc_feat, order=2, width=3)
#         spctrl_ctrd = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spctrl_bdwth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         spctrl_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         rms = librosa.feature.rms(y=y)
#         zcr = librosa.feature.zero_crossing_rate(y=y)

#         features.append(np.hstack([
#             np.mean(y), np.std(y), np.max(y), np.min(y),
#             np.mean(zcr), np.mean(rms),
#             np.mean(spctrl_ctrd), np.std(spctrl_ctrd),
#             np.mean(spctrl_bdwth), np.std(spctrl_bdwth),
#             np.mean(spctrl_rolloff), np.std(spctrl_rolloff),
#             np.mean(mfcc_feat, axis=1),       
#             np.std(mfcc_feat, axis=1),        
#             # np.mean(delta, axis=1),           
#             # np.std(delta, axis=1),            
#             # np.mean(delta2, axis=1),          
#             # np.std(delta2, axis=1),           
#             item["speaker"], item["digit"], item["sample_number"],
#         ]))
    
#     return pd.DataFrame(features)

class Augmentation:
    def __init__(self):
        pass
    
    # ==========================================
    # 1. PRE-MFCC (Waveform Augmentation)
    # Bekerja pada sinyal audio mentah (y)
    # ==========================================
    
    def add_gaussian_noise(self, audio_signal, noise_factor=0.005):
        """Menambahkan noise putih (Gaussian) ke sinyal audio."""
        noise = np.random.randn(len(audio_signal)).astype('float32')
        augmented = audio_signal + noise_factor * noise
        return augmented
    
    def time_shift(self, audio_signal, time_shift_max=0.2):
        """Menggeser sinyal secara horizontal (cyclic shift)."""
        shift = int(np.random.uniform(-time_shift_max, time_shift_max) * len(audio_signal))
        augmented = np.roll(audio_signal, shift)
        return augmented
    
    def pitch_shift(self, audio_signal, sr, pitch_steps=None):
        """Mengubah tinggi nada tanpa mengubah durasi."""
        # if pitch_steps is None:
        #     pitch_steps = [-2, -1, 1, 2]
        step = random.choice(pitch_steps)
        augmented = librosa.effects.pitch_shift(y=audio_signal, sr=sr, n_steps=step)
        return augmented
    
    def time_stretch(self, audio_signal, rate=None):
        """Mempercepat atau memperlambat durasi audio."""
        if rate is None:
            rate = random.uniform(rate[0], rate[1])
        augmented = librosa.effects.time_stretch(y=audio_signal, rate=rate)
        return augmented
    
    def volume_scale(self, audio_signal, scale_range=(0.7, 1.3)):
        """Mengubah gain/volume audio secara acak."""
        factor = random.uniform(scale_range[0], scale_range[1])
        augmented = audio_signal * factor
        return augmented

    def polarity_inversion(self, audio_signal):
        """Membalik fase gelombang audio (mengalikan dengan -1)."""
        return -audio_signal

    # ==========================================
    # 2. PASCA-MFCC (Spectral Augmentation)
    # Bekerja pada matriks MFCC (SpecAugment)
    # ==========================================

    def time_masking(self, mfcc, max_mask_t=5):
        """
        Menghapus blok kolom pada MFCC untuk simulasi data hilang.
        mfcc shape: [n_mfcc, time_steps]
        """

        n_mfcc, time_steps = mfcc.shape

        # Jika terlalu pendek, skip augmentation
        if time_steps <= 1:
            return mfcc

        # Pastikan mask tidak lebih besar dari dimensi waktu
        max_t = min(max_mask_t, time_steps - 1)

        if max_t <= 0:
            return mfcc

        # panjang mask
        t = np.random.randint(1, max_t + 1)

        # posisi awal mask
        t0 = np.random.randint(0, time_steps - t + 1)

        mfcc_aug = mfcc.copy()
        mfcc_aug[:, t0:t0+t] = 0

        return mfcc_aug


    def freq_masking(self, mfcc, max_mask_f=2):
        """
        Menghapus blok baris frekuensi pada MFCC.
        """

        n_mfcc, time_steps = mfcc.shape

        # Jika jumlah MFCC terlalu kecil
        if n_mfcc <= 1:
            return mfcc

        # Pastikan mask tidak melebihi jumlah frekuensi
        max_f = min(max_mask_f, n_mfcc - 1)

        if max_f <= 0:
            return mfcc

        # panjang mask
        f = np.random.randint(1, max_f + 1)

        # posisi awal mask
        f0 = np.random.randint(0, n_mfcc - f + 1)

        mfcc_aug = mfcc.copy()
        mfcc_aug[f0:f0+f, :] = 0

        return mfcc_aug

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