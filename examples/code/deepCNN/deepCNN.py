"""
Example membership inference attack against a deep net classifier on the CIFAR10 dataset
"""
import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D


import pickle
import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000
BATCH_SIZE=128

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 100, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 100, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 100, "Number of epochs to train attack models.")
train_target_model=True
target_model_filename=None
if(train_target_model==False and os.access(target_model_filename, os.R_OK)!=True):
    exit(1)#need to specify writable target_model_fiename

train_shadow_model=True
X_shadow_filename=None 
y_shadow_filename=None    
if(train_shadow_model==False and not(os.access(X_shadow_filename, os.R_OK)==True and os.access(y_shadow_filename, os.R_OK)==True ) ):
    exit(1)#need to specify writable shadow_model_fiename

train_attack_model=True
attack_model_filename=None    
if(train_attack_model==False and os.access(attack_model_filename, os.R_OK)!=True):
    exit(1)#need to specify writable attack_model_fiename


def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""
    
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    # initiate RMSprop optimizer
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model



def demo(argv):
    del argv  # Unused.

    (X_train, y_train), (X_test, y_test) = get_data()

    # Train the target model.
    
    target_model = target_model_fn()
    if(train_target_model==True):
        print("Training the target model...")
        target_model.fit(
            X_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.5, verbose=True,batch_size=BATCH_SIZE
        )
        target_model.save_weights('.target_model'+nowTime)
    else:
        print("load target model...")
        target_model.load_weights(target_model_filename)
    # Train the shadow models.
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    if(train_shadow_model==True):
        print("Training the shadow models...")
        X_shadow, y_shadow = smb.fit_transform(
            attacker_X_train,
            attacker_y_train,
            fit_kwargs=dict(
                batch_size=BATCH_SIZE,
                epochs=FLAGS.target_epochs,
                verbose=True,
                validation_data=(attacker_X_test, attacker_y_test),
            ),
        )
        output = open('X_shadow'+nowTime, 'wb')
        pickle.dump(X_shadow, output,-1)
        output.close()
        output = open('y_shadow'+nowTime, 'wb')
        pickle.dump(y_shadow, output,-1)
        output.close()
    else:
        print("load shadow model result...")
        output = open(X_shadow_filename, 'rb')
        X_shadow=pickle.load(output)
        output.close()
        output = open(y_shadow_filename, 'rb')
        y_shadow=pickle.load(output)
        output.close()
        

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    if(train_attack_model):
        print("Training the attack models...")
        amb.fit(
            X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True,batch_size=BATCH_SIZE)
        )
        output=open('amb'+nowTime, 'wb')
        pickle.dump(amb,output,-1)
        output.close()
    else:
        print("loading the attack models...")
        output = open(attack_model_filename, 'rb')
        amb=pickle.load(output)
        output.close()
        
        

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print(attack_accuracy)


if __name__ == "__main__":
    app.run(demo)
