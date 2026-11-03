import logging

def setup_logger(outdir:str ='.', label:str =None, log_level:str ='debug'):
    logger = logging.getLogger('Bilby-Extra')

    level = getattr(logging, log_level.upper())
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%m/%d/%y %H:%M')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    logger.setLevel(level)

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if label is not None:
        file_handler = logging.FileHandler(f'{outdir}/{label}.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
