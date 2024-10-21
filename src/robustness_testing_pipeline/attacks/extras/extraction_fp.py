import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.datasets import cifar10
import keras
import numpy as np
tf.compat.v1.disable_eager_execution()
from art.attacks.extraction.copycat_cnn import CopycatCNN
from art.estimators.classification import KerasClassifier
import numpy as np

target_model = load_model("models/trainedResnet_20241015_2232.h5")

target_classifier = KerasClassifier(model=target_model)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def create_theived_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

theived_model = create_theived_model()

theived_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

theived_classifier = KerasClassifier(model=theived_model)
 
copycat = CopycatCNN(target_classifier, 64, 64, 10, 100000)

new_class = copycat.extract(X_train, y_train, thieved_classifier = theived_classifier)

predictions = new_class.predict(X_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on test examples: {}%".format(accuracy * 100))



