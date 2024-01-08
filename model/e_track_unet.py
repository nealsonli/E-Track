# unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with unet.  If not, see <http://www.gnu.org/licenses/>.
import logging
import numpy as np
import tensorflow as tf
import unet
import timeit
from tensorflow.keras import backend as K
from unet import custom_objects, utils

import pupil_event_dataset


img_x_size = 352  # 346
img_y_size = 256  # 260
LEARNING_RATE = 1e-3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
np.random.seed(98765)


def weighted_categorical_crossentropy(weights):

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_wcc = y_true * K.log(y_pred) * weights
        loss_wcc = -K.sum(loss_wcc, -1)
        return loss_wcc

    return loss


def train():
    unet_model = unet.build_model(nx=img_x_size,
                                  ny=img_y_size,
                                  channels=3,
                                  num_classes=2,
                                  layer_depth=3,
                                  filters_root=32,
                                  padding="same"
                                  )

    unet.finalize_model(unet_model,
                        loss=weighted_categorical_crossentropy(np.array([0.1, 0.9])),
                        auc=False,
                        epsilon=0.000001,
                        learning_rate=LEARNING_RATE
                        )

    unet_model.summary()

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="mean_iou",
        min_delta=0,
        patience=3,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True
    )

    trainer = unet.Trainer(name="pupil_event", callbacks=[callback])

    users = range(4, 28, 1)
    train_tfrecs = np.array(tf.io.gfile.glob(f"./tfrecord_0/*user-{user}*.tfrec" for user in users))
    valid_tfrecs = np.array(tf.io.gfile.glob(f"./tfrecord_1/*user-{user}*.tfrec" for user in users))
    test_tfrecs = np.array(tf.io.gfile.glob(f"./tfrecord_2/*user-{user}*.tfrec" for user in users))
    train_dataset = pupil_event_dataset.load_data(train_tfrecs)
    valid_dataset = pupil_event_dataset.load_data(valid_tfrecs)
    test_dataset = pupil_event_dataset.load_data(test_tfrecs)

    for feat in train_dataset.take(1):
        print(f"Image shape: {feat[0].shape}")
        print(f"Label shape: {feat[1].shape}")
        print(f"Label shape: {feat[1].dtype}")

    print(f"Start Fit")
    trainer.fit(unet_model,
                train_dataset,
                valid_dataset,
                test_dataset,
                verbose=2,
                epochs=40,
                batch_size=8)
    print(f"End Fit")
    return unet_model


def predict():
    custom_objects['loss'] = weighted_categorical_crossentropy(np.array([0.1, 0.9]))
    unet_model = tf.keras.models.load_model('pupil_event/2023-01-24T00-11_42',
                                            custom_objects=custom_objects)
    unet_model.summary()

    users = range(4, 28, 1)
    test_tfrecs = np.array(tf.io.gfile.glob(f"./tfrecord_2/*user-{user}*.tfrec" for user in users))
    test_dataset = pupil_event_dataset.load_data(test_tfrecs)

    count = 0
    for element in test_dataset:
        count += 1
    warmupResult = unet_model.predict(tf.zeros((1, img_x_size, img_y_size, 3)))
    prediction = unet_model.predict(test_dataset.batch(batch_size=1))
    print(f'***** test_dataset len: {count}')

    results = unet_model.evaluate(test_dataset.take(3).batch(batch_size=1))
    print(results)

    test_dataset_3 = test_dataset.take(3)
    start_time = timeit.default_timer()
    prediction = unet_model.predict(test_dataset.batch(batch_size=1))
    prediction = unet_model.predict(test_dataset.batch(batch_size=1))
    prediction = unet_model.predict(test_dataset.batch(batch_size=1))
    prediction = unet_model.predict(test_dataset.batch(batch_size=1))
    run_time = timeit.default_timer() - start_time
    print(f'***** time: {run_time}')

    dataset = test_dataset.map(utils.crop_image_and_label_to_shape(prediction.shape[1:]))
    print(prediction.shape)


if __name__ == '__main__':
    train()

    with tf.device('/cpu:0'):
        predict()
