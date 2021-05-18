import h5py
import numpy as np
from os.path import exists


class DatasetRecorder:
    def __init__(self, filename: str, sampleLength: int,):
        self.filename: str = filename
        self.sampleLength: int = sampleLength
        self.currentGesture: str = None
        self.currentFrame: int = 0

        self.datasetHandle = None

        # initialize h5 file handle
        if exists(self.filename):
            self.file: h5py.File = h5py.File(self.filename, 'r+')
        else:
            self.file: h5py.File = h5py.File(self.filename, 'x')

    def preSampleRecord(self,) -> None:
        """
        Called before recording each sample, resets
        temporary parameters and counters, preps the
        current h5 dataset.
        """
        self.currentFrame = 0
        self.currentSampleCache = np.empty((self.sampleLength, 21, 3))

    def setCurrentGesture(self, label: str,) -> None:
        """
        Swaps the current gesture being recorded to
        a new one defined by `label`.

        Initializes the h5 dataset for this specific
        gesture, if it doesn't already exist.
        """
        self.currentGesture = label
        if self.currentGesture not in self.file:
            # We need to initialize the dataset
            self.datasetHandle = self.file.create_dataset(self.currentGesture, (1, self.sampleLength, 21, 3), maxshape=(
                None, self.sampleLength, 21, 3), dtype='float64')
        else:
            self.datasetHandle = self.file[self.currentGesture]

    def addSampleToDataset(self,) -> None:
        """
        Add the current sample to the dataset.
        """
        self.datasetHandle.resize(self.datasetHandle.shape[0]+1, axis=0)
        self.datasetHandle[-1] = np.array(self.currentSampleCache)

    def addFrameToSample(self, hand) -> None:
        """
        Add a frame of a hand to the current
        sample's cache.

        The frame is copied to the cache, and any
        further changes to the input array are not
        reflected in the cache.
        """
        self.currentSampleCache[self.currentFrame] = np.array(hand)
        self.currentFrame += 1
