import numpy as np
import librosa

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