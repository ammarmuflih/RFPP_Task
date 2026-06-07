import torch

class Config:
    # Path
    DATASET_PATH = "./datasets/FSDD"
    SAVE_DIR = "./results/models"
    AUDIO_PATH = "C:\\Users\\Muhammad Ammar M\\Nextcloud\\My Documents\\Kuylah S2\\SEM 2\\RFPP\\RFPP_Task\\Last Project\\pseudo MTL\\data\\recordings\\"
    SPECTOGRAM_PATH = "C:\\Users\\Ammar\\OneDrive\\Dokumen\\NextCloud\\My Documents\\Kuylah S2\\SEM 2\\RFPP\\RFPP_Task\\Last Project\\pseudo MTL\\data\\spectrograms\\"
    
    # Audio Feature Extraction
    SR = 22050
    N_MFCC = 13
    
    # Model Hyperparameters
    INPUT_SIZE = 45  # Sesuai jumlah fitur yang kamu gabung tadi
    HIDDEN_SIZE = 128
    LR = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # Multi-Task Weights
    WEIGHT_DIGIT = 1.0
    WEIGHT_SPEAKER = 1.5
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")