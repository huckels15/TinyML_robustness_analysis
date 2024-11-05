import os
import pickle
import numpy as np
import tensorflow as tf


def get_cifar10_test_ds_f32(negatives=False):
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/cifar-10-batches-py')

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_labels = np.array(cifar_test_labels)

    cifar_test_data = np.array(cifar_test_data / 255).astype(np.float32)
    cifar_test_labels = tf.one_hot(cifar_test_labels, 10).numpy()

    return cifar_test_data, cifar_test_labels


def get_cifar10_train_ds_f32(negatives=False):
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/cifar-10-batches-py')

    # Load all training batches
    cifar_train_data = []
    cifar_train_labels = []
    for i in range(1, 6):
        data_batch_dict = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        cifar_train_data.append(data_batch_dict[b'data'])
        cifar_train_labels += data_batch_dict[b'labels']

    # Concatenate all training batches
    cifar_train_data = np.vstack(cifar_train_data)
    cifar_train_labels = np.array(cifar_train_labels)

    # Reshape and process data
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    
    # Normalize data to range [0, 1]
    cifar_train_data = np.array(cifar_train_data / 255).astype(np.float32)
    
    # One-hot encode the labels
    cifar_train_labels = tf.one_hot(cifar_train_labels, 10).numpy()

    return cifar_train_data, cifar_train_labels

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def get_vww_test_ds_f32():
    data_test = []
    labels_test = []
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/vw_coco2014_96')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1. / 255)
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(96, 96),
        batch_size=1,
        subset='validation',
        color_mode='rgb',
        shuffle=False)

    batch_index = 0
    while batch_index < val_generator.n:
        data = next(val_generator) # changed this
        data_test.append(data[0][0])
        labels_test.append(np.argmax(data[1][0]))
        batch_index = batch_index + 1

    data_test = np.array(data_test)
    labels_test = tf.one_hot(labels_test, 2).numpy()

    return data_test, labels_test


def get_vww_train_ds_f32(batch_limit=625):  # Add a batch_limit parameter
    data_train = []
    labels_train = []
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/vw_coco2014_96')

    # ImageDataGenerator for training data
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,  # Use this only if part of the data is for validation
        rescale=1. / 255
    )

    # Generate training data from the training subset
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(96, 96),
        batch_size=32,
        subset='training',  # Use the 'training' subset
        color_mode='rgb',
        shuffle=False
    )

    batch_index = 0
    # Loop through the batches, stopping early once the batch_limit is reached
    while batch_index < batch_limit and batch_index < train_generator.n:
        data = next(train_generator)  # Get the next batch of images and labels
        data_train.append(data[0][0])  # Append image data
        labels_train.append(np.argmax(data[1][0]))  # Append the label (one-hot to class index)
        batch_index += 1

    data_train = np.array(data_train)
    labels_train = tf.one_hot(labels_train, 2).numpy()  # One-hot encode the labels

    return data_train, labels_train


def get_coffee_test_ds_f32():
    data_test = []
    labels_test = []

    test_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/test/')
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_gen = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=1, color_mode='rgb')

    batch_index = 0
    while batch_index < test_gen.n:
        data = test_gen.next()
        img = data[0][0] / 255
        data_test.append(img)
        labels_test.append(np.argmax(data[1][0]))
        batch_index = batch_index + 1

    data_test = np.array(data_test).astype(np.float32)
    labels_test = np.array(labels_test)
    labels_test = tf.one_hot(labels_test, 4).numpy()

    return data_test, labels_test