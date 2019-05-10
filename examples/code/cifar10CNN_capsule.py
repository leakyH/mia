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
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 50, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 50, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 10, "Number of epochs to train attack models.")

def squash(x, axis=-1):
    """The Squashing Function.
    The nonlinear activation function used in Capsule Network
    # Arguments
        x: Input Tensor.
        axis: Integer axis along which the squashing function is to be applied.
    # Returns
        Tensor with scaled value of the input tensor
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


def margin_loss(y_true, y_pred):
    """Margin loss
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

class Capsule(layers.Layer):
    """Capsule Network
    A Capsule Network Layer implementation in Keras
    There are two versions of Capsule Networks.
    One is similar to dense layer (for the fixed-shape input),
    and the other is similar to time distributed dense layer
    (for inputs of varied length).
    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )
    The Capsule implementation is from https://github.com/bojone/Capsule/
    # Arguments
        num_capsule: An integer, the number of capsules.
        dim_capsule: An integer, the dimensions of the capsule.
        routings: An integer, the number of routings.
        share_weights: A boolean, sets weight sharing between layers.
        activation: A string, the activation function to be applied.
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = int(input_shape[-1])
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = int(input_shape[-2])
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs, **kwargs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.
        This change can improve the feature representation of the capsule.
        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to get standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.int_shape(inputs)[0]
        input_num_capsule = K.int_shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        print(self.routings)
        for i in range(self.routings):
            c = K.softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):
        return None, self.num_capsule, self.dim_capsule


def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)
    return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""
    # A simple Conv2D model
    input_image = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(64, (3, 3), activation='relu')(input_image)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)

    # Now, we reshape it to (batch_size, input_num_capsule, input_dim_capsule)
    # then connect a capsule layer.
    # The output of final model is the lengths of 10 capsules, which have 16 dimensions.
    # The length of the output vector of the capsule expresses the probability of
    # existence of the entity, so the problem becomes a 10 two-classification problem.

    x = layers.Reshape((-1, 128))(x)
    capsule = Capsule(10, 16, 3, True)(x)
    output = layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
    model = Model(inputs=input_image, outputs=output)

    # Margin loss is used
    model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
    model.summary()
    

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
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        X_train, y_train, batch_size=128, epochs=FLAGS.target_epochs, validation_split=0.5, shuffle=True
    )
    #target_model.fit(
    #    X_train, y_train, batch_size=128, epochs=FLAGS.target_epochs, validation_data=(X_test, y_test), shuffle=True
    #)

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

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

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
