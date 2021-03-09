import tensorflow as tf
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D

batch_size=128
epochs=150
input_shape = (21,3)
classes = 6
x,y=[],[]

with h5py.File('gestures.hdf5','a') as f:
    for i,gesture in enumerate(list(f['/'])):
        print(i,gesture)
        data = list(f['/'+gesture+'/ds1'])
        x.extend(data)
        y.extend([i]*len(data))

x=np.array(x)
y=np.array(y) 
y=tf.keras.utils.to_categorical(y,classes)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,shuffle=True)
print(x_train.shape,x_test.shape)
    

model = tf.keras.models.Sequential([
    Flatten(input_shape=(21,3)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(classes, activation='softmax')
])
model.build()
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.1)

print(model.evaluate(x_test, y_test))
model.save("static_gestures_v1")