import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np

# Enable GPU instead of CPU by uncommenting below lines
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Variables (Top Half are Changeable Parameters)
labels = ["swipe_right", "swipe_left", "swipe_up", "swipe_down"] #All Gestures in order of ids from 0-x
gestures = ["swipe_right", "swipe_up"] #Gestures to pull from actual data set
data_parts = 5 #Number of scaled copies of dataset added on
filename = 'dynamic_gestures2.hdf5' #File Name of Gestures Location
reverse_time = {"swipe_right":"swipe_left", "swipe_up":"swipe_down"} #Gestures to reverse on time-axis to make new gesture
scale_low = .75 #Lower bound scale factor
scale_high = 1.25 #Upper bound scale factor
input_shape = (20,22,3) #Input shape of each gesture in data set
epochs = 10 #Number of Epochs for Training

labels = {label:i for i,label in enumerate(labels)}
classes = len(labels) 

def generate_translation(data):
    generated = np.empty([len(data),20,1,3])
    for i,sample in enumerate(data):
        last = np.zeros(3)
        for j,frame in enumerate(sample):
            d = np.array([(last-frame[0])])
            generated[i][j]=d
            last = frame[0]
    data = np.concatenate((data,generated),axis=2)
    return data
# Functions Used
x,y = np.empty([1,*list(input_shape)]), np.empty([0])
def addScaledData(label, data, amount): #Adds amount scaled copies of data to the final data set
    global x, y, input_shape, scale_low, scale_high
    m = len(data)
    n = input_shape[0] * input_shape[1] * input_shape[2]
    for _ in range(amount):
        mult = np.array([np.array([np.random.uniform(scale_low, scale_high)]*n).reshape(input_shape) for _ in range(m)])
        x = np.concatenate((x,generate_translation(data)*mult), axis=0)
    y = np.concatenate((y,np.array([labels[label]]*(amount*len(data)))), axis=0)

# Generate Data Set
print("Generating Dataset")
f = h5py.File(filename, 'r')
for gesture in gestures:
    data = f['/'+gesture]
    x = np.concatenate((x,generate_translation(data)), axis=0)
    y = np.concatenate((y,np.array([labels[gesture]]*len(data))), axis=0)
    addScaledData(gesture, data, data_parts)
    if gesture in reverse_time: #Check if we want to reverse the data on time-axis to get a different gesture
        gesture = reverse_time[gesture]
        data = np.array([np.flip(seg, axis=0) for seg in data])
        x = np.concatenate((x,generate_translation(data)), axis=0)
        y = np.concatenate((y,np.array([labels[gesture]]*len(data))), axis=0)
        addScaledData(gesture, data, data_parts)
x = x[1:]
y=tf.keras.utils.to_categorical(y,classes)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,shuffle=True)
print(x_train.shape, y_train.shape)

# Machine Learning Module Architecture
inputs = layers.Input(shape=input_shape)
x = inputs      
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(classes, activation='softmax',name='Output')(x)
model = keras.models.Model(inputs, output)

# Compile, Fit, and Save Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

model.fit(x=x_train, y=y_train, epochs=epochs, validation_data = (x_test, y_test))
model.save("test_model_2", save_format = "tf")