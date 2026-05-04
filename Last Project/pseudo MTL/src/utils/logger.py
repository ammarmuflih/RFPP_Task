import logging
import os

def setup_logger(log_dir="./results/logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger("PseudoMTL")
    logger.setLevel(logging.INFO)
    
    # Format: Waktu - Nama Logger - Level - Pesan
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler untuk simpan ke File
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(formatter)
    
    # Handler untuk tampil di Terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger