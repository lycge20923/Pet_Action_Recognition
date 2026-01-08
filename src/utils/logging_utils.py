import os
import logging
from datetime import datetime


def setup_logger(file_path: str, 
                 log_dir: str = "logs", 
                 level: int = logging.ERROR) -> logging.Logger:
    '''
    Input:
        file_path: the .py path
        log_dir: directory to save logs
        level: log level
    Output:
        logger: logger object
    '''
    
    # 1. set the file path to save log 
    os.makedirs(log_dir, exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = os.path.basename(file_path).split('.')[0]
    log_file = os.path.join(log_dir, f"{file_name}_{time}.log")

    # 2. get root logger, remove old handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. basic set
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    return root_logger
