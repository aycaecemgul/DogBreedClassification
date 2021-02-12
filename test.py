import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pickle
from matplotlib import pyplot as plt

IMG_SIZE=120
CATEGORIES=["Afghan Hound","Beagle","Blenheim Spaniel","Japanese Spaniel","Maltese Terrier",
            "Papillon","Pom","Rhodesian Ridgeback","Samoyed","Shih-Tzu"]

model=tf.keras.models.load_model("dog_classifier.model")

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

predict=model.predict([X_test])

t=10
p=np.argmax(predict[t])
q=y_test[t]

print("Actual breed of the dog: "+str(CATEGORIES[q]))
print("The prediction of the model: "+str(CATEGORIES[p]))

X_test[t]=X_test[t].reshape(IMG_SIZE,IMG_SIZE,3)

plt.imshow(X_test[t])


plt.show()
