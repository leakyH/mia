#论文算法复现

#所用库传入
import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

#所用参数设定
NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 100, "target model 和 shadow model的epoch数量"
)
flags.DEFINE_integer("attack_epochs", 100, "attack model的epoch数量")
flags.DEFINE_integer("num_shadows", 100, "shadow model的数量")

#cifar10数据输入
def get_data():
   
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)

#目标所用模型
def target_model_fn():
    """在论文中我们提到了可以通过调用API的方式实现shadow model模仿target model进行训练。
    target model实际上是一个黑盒，这里给出网络主要是探究成员推断攻击对于不同网络的效果。"""

    model = tf.keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

#攻击模型
def attack_model_fn():
    
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

#复现主程序
def demo(argv):
    del argv  .

    (x_train, y_train), (x_test, y_test) = get_data()

    # 训练target model
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        x_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.9, verbose=True
    )

    # 训练shadow model
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    # Shadow model所用数据
    attacker_x_train, attacker_x_test, attacker_y_train, attacker_y_test = train_test_split(
        x_test, y_test, test_size=0.1
    )
    print(attacker_x_train.shape, attacker_x_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_x_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_x_test, attacker_y_test),
        ),
    )

    # ShadowModelBundle 返回 AttackModelBundle 所需的数据形式
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # 训练attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

    # 测试attack model效果

    # 生成测试集
    data_in = x_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = x_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # 计算成员检测攻击的准确度
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print(attack_accuracy)


if __name__ == "__main__":
    app.run(demo)
