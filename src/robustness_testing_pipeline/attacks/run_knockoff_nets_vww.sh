#!/bin/bash
DATASET_ID=vww
DATASET_SIZE=1000
NUM_CLASSES=2
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

QNN_INT8='models/vww_96_20241021_1356_logits_quant.tflite'

#QNN_INT8='models-defense_enhanced/cifar_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_distil_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_ens_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_sat_int8.tflite'

echo "=============================================================="
echo "======================KnockoffNets=============================="
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --target_int8 "$QNN_INT8"