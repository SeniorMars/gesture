import h5py
import numpy as np
from os.path import exists


class DatasetRecorder:
    """
    Composes samples and saves them in dataset file.
    """

    def __init__(
        self,
        filename: str,
        sampleLength: int,
    ):
        self.filename: str = filename
        self.sampleLength: int = sampleLength
        self.currentGesture: str = None
        self.currentFrame: int = 0
        self.currentSampleCache = np.empty((self.sampleLength, 21, 3))

        self.datasetHandle = None

        # initialize h5 file handle
        if exists(self.filename):
            self.file: h5py.File = h5py.File(self.filename, "r+")
        else:
            self.file: h5py.File = h5py.File(self.filename, "x")

    def setCurrentGesture(
        self,
        label: str,
    ) -> None:
        """
        Swaps the current gesture being recorded to
        a new one defined by `label`.

        Initializes the h5 dataset for this specific
        gesture, if it doesn't already exist. Note that
        this creates an empty sample at position 0 of
        the dataset.
        """
        self.currentGesture = label
        if self.currentGesture not in self.file:
            # We need to initialize the dataset
            self.datasetHandle = self.file.create_dataset(
                self.currentGesture,
                (0, self.sampleLength, 21, 3),
                maxshape=(None, self.sampleLength, 21, 3),
                dtype="float64",
            )
        else:
            self.datasetHandle = self.file[self.currentGesture]

    def addSampleToDataset(
        self,
    ) -> None:
        """
        Add the current sample to the dataset, reset
        the caches and counters.
        """
        # if len(self.datasetHandle)==1 and np.count_nonzero(self.datasetHandle[0])>0:
        self.datasetHandle.resize(self.datasetHandle.shape[0] + 1, axis=0)
        self.datasetHandle[-1] = np.array(self.currentSampleCache)
        self.currentFrame = 0
        self.currentSampleCache = np.empty((self.sampleLength, 21, 3))

    def addFrameToSample(self, hand) -> None:
        """
        Add a frame of a hand to the current
        sample's cache.

        The frame is copied to the cache, and any
        further changes to the input array are not
        reflected in the cache.

        If the frame is null, this will copy the
        previous frame if there is one. Otherwise,
        it won't increment the counter, and won't
        add any frames.
        """
        if hand is None and self.currentFrame > 0:
            self.currentSampleCache[self.currentFrame] = np.array(
                self.currentSampleCache[self.currentFrame - 1]
            )
        else:
            self.currentSampleCache[self.currentFrame] = np.array(hand)
        self.currentFrame += 1
