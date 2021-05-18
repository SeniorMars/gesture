import h5py
import numpy as np
from os.path import exists

class DatasetRecorder:
    def __init__(self, filename: str, sampleLength: int,):
        self.filename = filename
        self.sampleLength = sampleLength

