import os
import pickle
import cv2 as cv
import numpy as np

from numpy import asarray
import PIL
from PIL import Image
import skimage
from skimage.transform import resize, rotate
from skimage.color import rgb2gray,gray2rgb
from skimage.filters import median,threshold_otsu,meijering
from skimage.exposure import equalize_adapthist
from matplotlib import pyplot as plt
from skimage.io import imread
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DIR="images"
IMG_SIZE = 120


CATEGORIES=["Afghan Hound","Beagle","Blenheim Spaniel","Japanese Spaniel","Maltese Terrier",
            "Papillon","Pom","Rhodesian Ridgeback","Samoyed","Shih-Tzu"]

#FIXING IMAGE SIZE TO 120x120
def fix_image_size(DIR,IMG_SIZE):
    for category in os.listdir(DIR):
        path = os.path.join(DIR, category)
        for img in os.listdir(path):
            filename = os.path.join(path, img)
            img_array = asarray(Image.open(filename).convert('RGB'))
            img_array = resize(img_array, (IMG_SIZE, IMG_SIZE))
            plt.imsave(filename,img_array)

#counts to total number of images
def count_images(DIR):
    i=0
    for category in os.listdir(DIR):
        path = os.path.join(DIR, category)
        for img in os.listdir(path):
          i+=1

    print("amount of images= "+str(i))

# training_data=[]
#
# for category in os.listdir(DIR):
#     path = os.path.join(DIR, category)
#     for img in os.listdir(path):
#         class_no = os.listdir(DIR).index(category)
#         filename = os.path.join(path, img)
#         img_array = asarray(skimage.io.imread(filename))
#         training_data.append([img_array, class_no])
#
# print("training data is loaded.")
#
# #data augmentation
# for category in os.listdir(DIR):
#     path = os.path.join(DIR, category)
#     for img in os.listdir(path):
#         class_no = os.listdir(DIR).index(category)
#         filename = os.path.join(path, img)
#         img_array = asarray(skimage.io.imread(filename))
#         training_data.append([rotate(img_array, angle=30, resize=False), class_no])
#         training_data.append([cv.flip(img_array, 1), class_no])
#
# print("Data augmentation is done.")
#
# X=[]
# y=[]
#
# for image,label in training_data:
#     X.append(image)
#     y.append(label)
#
# X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)/255.0
# y = np.array(y)
#
# print(X.shape)
#
# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()
#
# print("Pickled.")

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

print("Pickle jar opened.")

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle= True)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.33, shuffle= True )

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

base_model = InceptionV3(
                                weights='imagenet',
                                include_top=False,
                                input_shape=(IMG_SIZE, IMG_SIZE,3)
                                )

base_model.trainable=False

model = tf.keras.Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())


model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

aug = ImageDataGenerator()

model.fit(aug.flow(X_train, y_train),validation_data = (X_val, y_val),epochs = 15,verbose=2)


model.save("dog_classifier.model")