import tensorflow as tf
from keras.applications.densenet import layers
from keras.layers import BatchNormalization
from keras.legacy_tf_layers.convolutional import Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics
from tensorflow.python.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import Reshape

train_data = 'YOUR_TRANING_DATA_PATH'
test_data = 'YOUR_TESTING_DATA_PATH'

train = ImageDataGenerator(rescale= 1 / 255)
test = ImageDataGenerator(rescale= 1 / 255)

train_generator = train.flow_from_directory(
    train_data,
    target_size=(28,28),
    color_mode = 'grayscale',
    batch_size=32
)

test_generator = test.flow_from_directory(
    test_data,
    target_size= (28,28),
    color_mode='grayscale',
    batch_size=32
)

inputs = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, 3, activation = 'relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = layers.Flatten()(x)
x = Reshape((1, -1))(x)

x = layers.LSTM(64, 'relu', )(x)

x = layers.Dense(64, 'relu',)(x)
outputs = layers.Dense(7, 'softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss = keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

model.fit(test_generator, batch_size=32, epochs=30, verbose = 1, validation_data=test_generator)

model.evaluate(test_generator)

model.save('LSTM_CNN.h5')


