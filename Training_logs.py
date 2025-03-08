import os
import logging

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(f"{log_dir}/training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(file_handler)
    return logger
