from gestures import GESTURES
from tensorflow import keras
import numpy as np
import h5py

# A dictionary detailing gesture information.


class DataGenerator(keras.utils.Sequence):
    def __init__(self, filename, batchSize=32, dim=(20, 21, 3)):
        # Open file
        self.file = h5py.File(filename, "r")
        self.batchSize = batchSize
        self.numClasses = len(GESTURES)
        self.dim = dim
        # Fetch data from h5py file
        for _, gesture in enumerate(list(self.file)):
            if gesture in GESTURES:
                data = list(self.file[gesture])
                GESTURES[gesture]["data"] = np.array(data)
        # How many samples exist in total (including to-be generated ones)
        self.totalDataAmount = 0
        # A list of indices that can be used for shuffling on each epoch.
        # Each element is a tuple of (index, gestureType), where index is the specific index of the sample to be used.
        self.indices = []
        self.idToName = {}
        # Populate self.indexes and self.idToName
        for name, props in GESTURES.items():
            id = props["id"]
            self.idToName[id] = name
            if "src" in props:
                props = GESTURES[props["src"]]

            l = len(props["data"])
            self.totalDataAmount += l
            self.indices.extend(list(enumerate([id] * l)))
        self.indices = np.array(self.indices)
        self.on_epoch_end()

    def __len__(self):
        return int(self.totalDataAmount / self.batchSize) * 3

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def center_sample(sample):
        return np.subtract(sample, sample[0][0])

    def __fetch_sample(self, gesture, index):
        reverse = False
        props = GESTURES[gesture]
        if "src" in props:
            props = GESTURES[props["src"]]
            reverse = True
        data = np.array(props["data"][index])
        if reverse:
            data = np.flip(data, axis=0)
        # Center the sample on the first frame's wrist.
        data = DataGenerator.center_sample(data)
        if props["scale"]:
            data = data * np.random.uniform(0.5, 1.5)
        if self.dim[0]==10:
            data = data[::2]
        return data

    def __data_generation(self, batchIndex):
        X = np.ndarray((self.batchSize, *self.dim))
        y = np.zeros(self.batchSize)
        for i in range(self.batchSize):
            index = batchIndex + i
            gesture = self.idToName[self.indices[index][1]]
            sampleIndex = self.indices[index][0]
            X[i] = self.__fetch_sample(gesture, sampleIndex)
            y[i] = self.indices[index][1]
        y = keras.utils.to_categorical(y, num_classes=self.numClasses)
        return X, y

    def __getitem__(self, batchIndex):
        batchIndex = (batchIndex * self.batchSize) % len(self.indices)
        if batchIndex + self.batchSize >= len(self.indices):
            batchIndex = len(self.indices) - self.batchSize
        return self.__data_generation(batchIndex)
