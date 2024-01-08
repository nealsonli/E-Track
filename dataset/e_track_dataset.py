import tensorflow as tf

# Create tf.data.Dataset from TFRecord files
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False
img_x_size = 346
img_y_size = 260
IMAGE_SIZE = [img_x_size, img_y_size]
paddings = tf.constant([[3, 3], [14, 14],  [0, 0]])
paddings_2d = tf.constant([[3, 3], [14, 14]])
target_img_x_size = 352
target_img_y_size = 256
print_figure = False


def _read_tfrecord(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    single_example = tf.io.parse_single_example(example, features)

    # --- IMAGE ------------------------------------------------------------
    image = tf.io.decode_raw(single_example['image'], out_type='uint8')
    image = tf.reshape(image, (img_x_size, img_y_size, 3))
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize_with_crop_or_pad(image, target_img_x_size, target_img_y_size)

    # --- LABEL ------------------------------------------------------------
    label = tf.io.decode_raw(single_example['label'], out_type='bool')
    # 1 Channel
    label = tf.reshape(label, (img_x_size, img_y_size, 1))
    label = tf.image.resize_with_crop_or_pad(label, target_img_x_size, target_img_y_size)
    # 2 Channel
    label = tf.concat([tf.math.logical_not(label), label], axis=2)

    label = tf.cast(label, tf.float32)
    return image, label


def load_data(tfrecs):
    dataset = tf.data.TFRecordDataset(tfrecs, compression_type="GZIP", num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_read_tfrecord)
    return dataset
