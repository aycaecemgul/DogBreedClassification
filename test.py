import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pickle

CATEGORIES=["Afghan Hound","Beagle","Blenheim Spaniel","Japanese Spaniel","Maltese Terrier",
            "Papillon","Pom","Rhodesian Ridgeback","Samoyed","Shih-Tzu"]

model=tf.keras.models.load_model("dog_classifier.model")

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

predict=model.predict([X_test])


p=np.argmax(predict[15])

q=y_test[15]

print("Actual breed of the dog: "+str(CATEGORIES[q]))
print("The prediction of the model: "+str(CATEGORIES[p]))

