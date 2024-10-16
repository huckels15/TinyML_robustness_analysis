#!/bin/bash
DATASET_ID=cifar10
DATASET_SIZE=1000
NUM_CLASSES=10
#ANN_FLOAT='models/cifar_resnet_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_adv_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_distil_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_ens_adv_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_sat_float.h5'
#QNN_INT16='models/cifar_resnet_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_adv_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_distil_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_ens_adv_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_sat_int16.tflite'

#QNN_INT8='models/pretrainedResnet_quant.tflite'
QNN_INT8='models/trainedResnet_20241016_0827_quant.tflite'

#QNN_INT8='models-defense_enhanced/cifar_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_distil_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_ens_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_sat_int8.tflite'

echo "=============================================================="
echo "======================BOUNDARY-INT8==========================="
python3 boundary_int8.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8" \
                        --logs_folder 'logs/cifar/boundary_int8' --boundary_delta 0.1 --boundary_epsilon 1.0 --boundary_max_iter 500

# echo "=============================================================="
# echo "======================GEODA-INT8=============================="
# python3 geoda_int8.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8" \
#                     --logs_folder 'logs/cifar/geoda_int8' --geoda_bin_search_tol 0.0001 --geoda_sub_dim 75