import os
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

class dataUtils:
    def __init__(self):
        pass

    def load_data(self, audio_dir):
        data = []
        for items in os.listdir(audio_dir):
            if not items.endswith(".wav"): continue
            
            audio_path = os.path.join(audio_dir, items)
            name = items.replace(".wav", "")
            
            try:
                digit, speaker_name, sample_number = name.split("_")
            except ValueError: continue

            data.append({
                "path": audio_path, # Simpan path saja
                "speaker": speaker_name,
                "digit": int(digit),
                "sample_number": int(sample_number)
            })
        return pd.DataFrame(data)
    
    def data_preprocess(self, train_df, test_df):
        scaler = StandardScaler()
        le = LabelEncoder()

        y_speaker_train = le.fit_transform(train_df[52])
        y_speaker_test = le.transform(test_df[52])

        y_digit_train = train_df[53].values
        y_digit_test = test_df[53].values

        X_train_raw = train_df.drop(columns=[52, 53]).values
        X_test_raw = test_df.drop(columns=[52, 53]).values

        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        return X_train_scaled, X_test_scaled, y_digit_train, y_digit_test, y_speaker_train, y_speaker_test

    def data_split(self, df):
        # Official Split: sample 0-4 (test), 5-49 (train)
        test_data = df[df['sample_number'].isin(range(0, 5))].reset_index(drop=True)
        train_data = df[df['sample_number'].isin(range(5, 50))].reset_index(drop=True)
        
        return train_data, test_data

class FSDDDataset(Dataset):
    def __init__(self, df, audio_transform=None, mfcc_transform=None):
        self.df = df
        self.audio_transform = audio_transform
        self.mfcc_transform = mfcc_transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load audio mentah (Pra-MFCC)
        y, sr = librosa.load(row['path'])
        
        # 2. Eksekusi Audio Augmentation (Sinyal Suara)
        if self.audio_transform:
            for aug_func in self.audio_transform:
                # Karena fungsi kamu di class Augmentation butuh parameter spesifik,
                # di sini kita panggil secara berurutan.
                y = aug_func(y, sr) if 'sr' in aug_func.__code__.co_varnames else aug_func(y)

        # 3. Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1400)

        # 4. Eksekusi MFCC Augmentation (SpecAugment)
        if self.mfcc_transform:
            for aug_func in self.mfcc_transform:
                mfcc = aug_func(mfcc)

        # 5. Pooling untuk MLP (Statistik rata-rata)
        mfcc_final = np.mean(mfcc.T, axis=0) 

        return (torch.FloatTensor(mfcc_final), 
                torch.tensor(row['digit']), 
                torch.tensor(row['speaker_id']))

    def __len__(self):
        return len(self.df)

if __name__=='__main__':
    audio_dir = 'C:\\Users\\Ammar\\OneDrive\\Dokumen\\NextCloud\\My Documents\\Kuylah S2\\SEM 2\\RFPP\\RFPP_Task\\Last Project\\pseudo MTL\\data\\recordings\\'
    dataUtils.load_data(audio_dir)