import logging

def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(f"{log_dir}/training.log")
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger
