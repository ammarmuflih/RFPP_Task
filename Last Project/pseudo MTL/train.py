# import argparse
# import yaml
from src.utils.logger import setup_logger
from src.data.dataloader import load_data
from src.utils.config import Config
# from src.features.pipeline import FeaturePipeline
# from src.models.mlp import MultiTaskMLP
# from src.models.trainer import Trainer

def main():
    data = load_data(Config.AUDIO_PATH)
    print(data)

    # # 2. Feature Pipeline
    # # Menginisialisasi ekstraksi MFCC, Spectral, dll
    # pipeline = FeaturePipeline(config['features'])

    # # 3. Data Loaders
    # # Mengasumsikan loader.py mengembalikan DataLoader PyTorch
    # train_loader = get_dataloader(config['data']['train_json'], pipeline, config['batch_size'], shuffle=True)
    # val_loader = get_dataloader(config['data']['val_json'], pipeline, config['batch_size'], shuffle=False)

    # # 4. Initialize Model
    # # input_dim didapat dari hasil ekstraksi fitur
    # model = MultiTaskMLP(
    #     input_dim=config['model']['input_dim'],
    #     num_digits=10, 
    #     num_speakers=config['model']['num_speakers']
    # )

    # # 5. Training Loop
    # trainer = Trainer(model, config, logger)
    # trainer.fit(train_loader, val_loader)

    # # 6. Save Model
    # model_path = f"{config['model_dir']}/final_model.pth"
    # trainer.save_checkpoint(model_path)
    # logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True, help="Path ke file .yaml")
    # args = parser.parse_args()
    main()