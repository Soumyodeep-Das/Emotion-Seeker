import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model # functional API
from keras.layers import Dense, Flatten

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# working with pre trained model

base_model = MobileNet(include_top=False, input_shape=(224, 224, 3)) # weight

for layer in base_model.layers: # to prevent retraning of the model
    layer.trainable = False # every layer trainability is false

x = Flatten()(base_model.output) # flatten the output of the base model
# add a dense layer
x = Dense(units=7 , activation='softmax' )(x)

# creating our model

model = Model(inputs=base_model.input, outputs=x)

#all the layers of the model

model.summary()

# compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# prepare data using ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255)

train_data = train_datagen.flow_from_directory(directory='dataset/train', target_size=(224, 224), batch_size=32)

train_data.class_indices


val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    directory='dataset/val', 
    target_size=(224, 224), 
    batch_size=32
    )

# visualize the data

t_img, label = train_data.next()

#function when called will plot the image
def plot_img(img_arr, label):
    # input - images array
    # output - plot the image 
    count = 0
    for im in img_arr:
        plt.imshow(im)
        plt.title(label[count])
        plt.axis(False)
        plt.show()
        count += 1

        if count == 10:
            break

# function call to plot the image
plot_img(t_img, label)

# having early stopping and model checkpoint

from keras.callbacks import EarlyStopping, ModelCheckpoint

# early stopping

es = EarlyStopping(monitor='val_accuracy', 
                   min_delta=0.01, 
                   patience=5, 
                   verbose=1, 
                   mode='auto'
                   )

# model checkpoint

mc= ModelCheckpoint(filepath='best_model.h5', 
                    monitor='val_accuracy', 
                    verbose=1, 
                    save_best_only=True, 
                    mode='auto')

# putting call back in a list

call_back = [es, mc]


hist = model.fit_generator(generator=train_data, 
                           steps_per_epoch=10, 
                           epochs=30, 
                           validation_data=val_data, 
                           validation_steps=8, 
                           callbacks=call_back
                           )

# loading the best fit model

from keras.models import load_model
model = load_model('best_model.h5')

h = hist.history
h.keys()

# plot the graph for accuracy
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c='red')
plt.title('acc vs v-acc')
plt.show()

# plot the graph for loss
plt.plot(h['loss'])
plt.plot(h['val_loss'], c='red')
plt.title('loss vs v-loss')
plt.show()

# just to map o/p values
op = dict(zip(train_data.class_indices.values(), train_data.class_indices.keys()))

# path for the images to see if it preditcs correctly or not

path = 'dataset/download.jpg'  # path for the image
img = load_img(path, target_size=(224, 224)) # load the image

i = img_to_array(img)/255.0 # convert the image to array and normalize it
input_arr = np.array([i]) # create a batch of 1 image
input_arr.shape

pred = np.argmax(model.predict(input_arr)) # predict the image

print(f"Predicted output is {op[pred]}")

# display the image

plt.imshow(input_arr[0])
plt.title("Input Image")
plt.show()