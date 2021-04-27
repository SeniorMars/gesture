import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py

gestures = {
    "swipe_up":
    {
        "id": 0,
        "scale": True,
        "fliplr": False,
        "flipud": False
    },
    "swipe_down":
    {
        "id": 1,
        "src": "swipe_up",
        "scale": True,
        "fliplr": False,
        "flipud": False
    },
    "swipe_right":
    {
        "id": 2,
        "scale": True,
        "fliplr": False,
        "flipud": False
    },
    "swipe_left":
    {
        "id": 3,
        "src": "swipe_right",
        "scale": True,
        "fliplr": False,
        "flipud": False
    },
}


class DataGenerator(keras.utils.Sequence):

    def __init__(self, filename, batch_size=32, num_classes=4, dim=(20, 21, 3)):
        # Open file
        self.file = h5py.File(filename, 'r')
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dim = dim
        # Fetch data from h5py file
        for _, gesture in enumerate(list(self.file['/'])):
            if gesture in gestures:
                data = list(self.file['/'+gesture])
                gestures[gesture]["data"] = np.array(data)
        # How many samples exist in total (including to-be generated ones)
        self.data_amt = 0
        # A list of indices that can be used for shuffling on each epoch
        self.indices = []
        self.counters = [[]]*self.num_classes
        self.ids_to_name = {}
        # Populate self.indexes
        for name, props in gestures.items():
            id = props["id"]
            self.ids_to_name[id] = name
            if "src" in props:
                props = gestures[props["src"]]

            l = len(props["data"])
            self.data_amt += l
            self.indices.extend(list(enumerate([id]*l)))
        self.indices = np.array(self.indices)
        self.on_epoch_end()

    def __len__(self):
        return int(self.data_amt/self.batch_size)*3

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __fetch_sample(self, gesture, index):
        reverse = False
        props = gestures[gesture]
        if "src" in props:
            props = gestures[props["src"]]
            reverse = True
        data = np.array(props["data"][index])
        if reverse:
           data = np.flip(data, axis=0)
        if props["scale"]:
            data = data*np.random.uniform(0.5, 1.5)
        return data

    def __data_generation(self, batch_index):
        X = np.ndarray((self.batch_size, *self.dim))
        y = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            index = batch_index+i
            gesture = self.ids_to_name[self.indices[index][1]]
            sub_index = self.indices[index][0]
            X[i] = self.__fetch_sample(gesture, sub_index)
            y[i] = self.indices[index][1]
        y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        return X, y

    def __getitem__(self, batch_index):
        batch_index = (batch_index * self.batch_size) % len(self.indices)
        if batch_index+self.batch_size >= len(self.indices):
            batch_index = len(self.indices)-self.batch_size
        return self.__data_generation(batch_index)
