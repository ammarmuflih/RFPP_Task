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
    
    def scale_features(train_df, val_df=None, test_df=None, feature_col='mfcc_feature'):
        scaler = StandardScaler()

        # Fit + transform hanya dari train
        X_train = np.vstack(train_df[feature_col].values)
        X_train_scaled = scaler.fit_transform(X_train)

        result = {'train': X_train_scaled, 'scaler': scaler}

        # Transform val & test pakai scaler dari train
        if val_df is not None:
            X_val = np.vstack(val_df[feature_col].values)
            result['val'] = scaler.transform(X_val)

        if test_df is not None:
            X_test = np.vstack(test_df[feature_col].values)
            result['test'] = scaler.transform(X_test)

        return result

    # def data_split(self, df):
    #     # Official Split: sample 0-4 (test), 5-49 (train)
    #     test_data = df[df['sample_number'].isin(range(0, 5))].reset_index(drop=True)
    #     train_data = df[df['sample_number'].isin(range(15, 50))].reset_index(drop=True)
    #     val_data = df[df['sample_number'].isin(range(5, 15))].reset_index(drop=True)
        
    #     return train_data, test_data, val_data

    def data_split(self, df):
        # Official Split: sample 0-4 (test), 5-49 (train)
        test_data = df[
            (df['sample_number'].isin(range(0, 5))) & 
            (df['augmented'] == 0)
        ].reset_index(drop=True)
        
        train_data = df[df['sample_number'].isin(range(15, 50))].reset_index(drop=True)
        
        val_data = df[
            (df['sample_number'].isin(range(5, 15))) & 
            (df['augmented'] == 0)
        ].reset_index(drop=True)

        return train_data, test_data, val_data
    
    def load_data_waveform(self, audio_dir):
        waveform_data = []
        for item in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, item)
            y, sr = librosa.load(audio_path)
            y, _ = librosa.effects.trim(y)
            name = item.replace(".wav", "")
            try:
                digit, speaker_name, sample_number = name.split("_")
            except ValueError: continue

            waveform_data.append({
                "data_path": audio_path,
                "y": y,
                "sr": sr,
                "augmented": 0,
                "speaker": speaker_name,
                "digit": digit,
                "sample_number": sample_number,
            })
        
        return waveform_data

class FSDDDataset(Dataset):
    def __init__(self, df, scaled_data, mfcc_transform=None):
        self.df = df
        self.scaled_data = scaled_data 
        self.mfcc_transform = mfcc_transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mfcc = self.scaled_data[idx]

        return (torch.FloatTensor(mfcc), 
                torch.tensor(row['digit']), 
                torch.tensor(row['speaker_id']))

    def __len__(self):
        return len(self.df)

if __name__=='__main__':
    audio_dir = 'C:\\Users\\Ammar\\OneDrive\\Dokumen\\NextCloud\\My Documents\\Kuylah S2\\SEM 2\\RFPP\\RFPP_Task\\Last Project\\pseudo MTL\\data\\recordings\\'
    dataUtils.load_data(audio_dir)