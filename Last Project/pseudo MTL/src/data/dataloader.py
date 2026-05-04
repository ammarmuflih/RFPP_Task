import os
import librosa
import pandas as pd

def load_data(audio_dir):
    data = []

    for items in os.listdir(audio_dir):
        # skip kalau bukan file .wav
        if not items.endswith(".wav"):
            continue

        audio_path = os.path.join(audio_dir, items)

        # load audio
        y, sr = librosa.load(audio_path)
        y, _ = librosa.effects.trim(y)

        # parsing nama file FSDD: digit_speaker_index.wav
        name = items.replace(".wav", "")
        try:
            digit, speaker_name, sample_number = name.split("_")
        except ValueError:
            # skip kalau format tidak sesuai
            continue

        # convert ke tipe yang benar
        digit = int(digit)
        sample_number = int(sample_number)

        data.append({
            "filename": items,
            "speaker": speaker_name,
            "digit": digit,
            "sample_number": sample_number,
            "y": y,
            "sr": sr,
        })

    return data

# def load_data(audio_dir):
#     data = []
#     for items in os.listdir(audio_dir):
#         audio_path = os.path.join(audio_dir, items)
#         y, sr = librosa.load(audio_path)
#         y, _ = librosa.effects.trim(y)
#         digit = items[0]

#         start_name = 0
#         stop_name = 0
#         sep_num = 0
#         dot_position = 0

#         for n, character in enumerate(items):
#             if character == "_" and sep_num == 0:
#                 start_name = n+1
#                 sep_num += 1

#             elif character == "_" and sep_num == 1:
#                 stop_name = n
#                 sep_num += 1

#             elif character == "." and sep_num == 2:
#                 dot_position = n
        
#         speaker_name = items[start_name:stop_name]
#         sample_number = items[stop_name+1:dot_position]
#         data.append({
#             "filename": items,
#             "speaker": speaker_name,
#             "digit": digit,
#             "sample_number": sample_number,
#             "y": y,
#             "sr": sr,
#         })
        
#     return data


if __name__=='__main__':
    audio_dir = 'C:\\Users\\Ammar\\OneDrive\\Dokumen\\NextCloud\\My Documents\\Kuylah S2\\SEM 2\\RFPP\\RFPP_Task\\Last Project\\pseudo MTL\\data\\recordings\\'
    load_data(audio_dir)