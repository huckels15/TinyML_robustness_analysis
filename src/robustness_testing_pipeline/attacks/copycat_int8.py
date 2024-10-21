import tensorflow as tf
from argparse import ArgumentParser
import utils.backend as b
import utils.dataset_loader as ds
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers.legacy import Adam
from tflite_to_cmsis import tflite_to_cmsis_main as cm
from art.attacks.extraction.copycat_cnn_int8 import CopycatCNN_Int8
from art.estimators.classification.tensorflow_int8 import TensorFlowV2Classifier_Int8
from model_converter import representative_dataset_generator, convert_model
from convert_vww import run_conversion
from art.estimators.classification.tensorflow import TensorFlowV2Classifier


def create_theived_model_cifar():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def create_theived_model_vww():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def load_configs():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dataset_id", type=str, default=None, help="dataset to use: cifar10 or vww")
    parser.add_argument("--batch_size_fit", type=int, default=64, help="number of samples to perturb")
    parser.add_argument("--batch_size_query", type=str, default=64, help="path to save the adversarial examples")
    parser.add_argument("--nb_epochs", type=float, default=100, help="")
    parser.add_argument("--nb_stolen", type=float, default=1000, help="")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes in target models")
    parser.add_argument("--target_int8", type=str, default=None, help="path to the int-8 QNN")
    parser.add_argument("--model_folder", type=str, default="/models", help="path to save thieved model")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'num_classes', 'target_int8']
    for arg_name in required_cfgs:
        if getattr(cfgs, arg_name) is None:
            raise ValueError(f"Required argument {arg_name} is not set.")
        
    run_cfgs = vars(cfgs)

    return run_cfgs


@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def main():
    # Step 1: Load configs
    cfgs = load_configs()

    if cfgs['dataset_id'] == 'cifar10':
        input_shape = (32,32,3)
        thieved_model = create_theived_model_cifar()
    elif cfgs['dataset_id'] == 'vww':
        input_shape = (96,96,3)
        thieved_model = create_theived_model_vww()

    # Step 2: Load ANN/QNNs
    qnn_int8 = b.get_ml_quant_model(cfgs['target_int8'])
    scaler_int8, zp_int8 = b.get_input_quant_details(qnn_int8)
    art_classifier = TensorFlowV2Classifier_Int8(model=qnn_int8, clip_values=(0, 1),
                            nb_classes=cfgs['num_classes'], input_shape=input_shape,
				            loss_object=tf.keras.losses.CategoricalCrossentropy())

    # Step 3: Load dataset and generate .bin files
    if cfgs['dataset_id'] == 'cifar10':
        x_train_float, y_train = ds.get_cifar10_train_ds_f32()
        x_test_float, y_test = ds.get_cifar10_test_ds_f32()
    elif cfgs['dataset_id'] == 'vww':
        # x_train_float, y_train = ds.get_vww_train_ds_f32() # PUT THIS BACK
        x_test_float, y_test = ds.get_vww_test_ds_f32()

    # x_test_float, y_test = x_test_float[0:1000], y_test[0:1000]
    # x_train_int8 = b.quantize_dataset_int8(x_train_float, scaler_int8, zp_int8) # PUT THIS BACK
    x_test_int8 = b.quantize_dataset_int8(x_test_float, scaler_int8, zp_int8)

    # Step 4: Evaluate classifier test examples
    accuracy = b.get_accuracy_quant_model(qnn_int8, x_test_int8, y_test)
    print("Int8 -> Accuracy on test examples: {}%".format(accuracy * 100) + "\n")

    # Step 5: Steal model
    attack = CopycatCNN_Int8(classifier=art_classifier, batch_size_fit=cfgs['batch_size_fit'],\
    batch_size_query=cfgs['batch_size_query'], nb_epochs=cfgs['nb_epochs'], nb_stolen=cfgs['nb_stolen'])
    thieved_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    thieved_classifier = TensorFlowV2Classifier(model=thieved_model, nb_classes=cfgs['num_classes'], input_shape=input_shape, train_step=train_step)
    stolen_model = attack.extract(x_test_float, y_test, thieved_classifier=thieved_classifier) # Change this back to train
    stolen_model.model.save("models/stolen_model.h5")

    if cfgs['dataset_id'] == 'cifar10':
        convert_model("stolen_model.h5")
    elif cfgs['dataset_id'] == 'vww':
        run_conversion("stolen_model.h5")


    stolen_int8 = b.get_ml_quant_model("models/stolen_model_quant.tflite")

    # Step 6: Evaluate the classifiers
    accuracy = b.get_accuracy_quant_model(stolen_int8, x_test_int8, y_test)
    print("Int8 -> Accuracy: {}%".format(accuracy * 100))


if __name__ == '__main__':
    main()