#!/bin/bash
DATASET_ID=cifar10
BASIC=basic
VGG=vgg
LENET=lenet
ALEXNET=alexnet
RESNET=resnet
DATASET_SIZE=1000
NUM_CLASSES=10
NB_EPOCS_100=1 #NB_EPOCS_100=100
NB_EPOCS_80=80
NB_STOLEN=10 #NB_STOLEN=50000

QNN_INT8='src/robustness_testing_pipeline/models/trainedResnet_testable_quant.tflite'

echo "=============================================================="
echo "======================CopycatCNN -- Basic 50000 100 =============================="
echo "Basic Arch Results:" >> copycat_cifar_results_50000_100_2.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$BASIC" >> copycat_cifar_results_50000_100_2.txt
echo "" >> copycat_cifar_results_50000_100_2.txt

echo "=============================================================="
echo "======================CopycatCNN -- Basic 50000 80 =============================="
echo "Basic Arch Results:" >> copycat_cifar_results_50000_80.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$BASIC" >> copycat_cifar_results_50000_80.txt
echo "" >> copycat_cifar_results_50000_80.txt

echo "=============================================================="
echo "======================CopycatCNN -- Lenet 50000 100 =============================="
echo "Lenet Arch Results:" >> copycat_cifar_results_50000_100_2.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$LENET" >> copycat_cifar_results_50000_100_2.txt
echo "" >> copycat_cifar_results_50000_100_2.txt

echo "=============================================================="
echo "======================CopycatCNN -- Lenet 50000 80 =============================="
echo "Lenet Arch Results:" >> copycat_cifar_results_50000_80.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$LENET" >> copycat_cifar_results_50000_80.txt
echo "" >> copycat_cifar_results_50000_80.txt

echo "=============================================================="
echo "======================CopycatCNN -- Alexnet 50000 100 =============================="
echo "Alexnet Arch Results:" >> copycat_cifar_results_50000_100_2.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$ALEXNET" >> copycat_cifar_results_50000_100_2.txt
echo "" >> copycat_cifar_results_50000_100_2.txt

echo "=============================================================="
echo "======================CopycatCNN -- Alexnet 50000 80 =============================="
echo "Alexnet Arch Results:" >> copycat_cifar_results_50000_80.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$ALEXNET" >> copycat_cifar_results_50000_80.txt
echo "" >> copycat_cifar_results_50000_80.txt

echo "=============================================================="
echo "======================CopycatCNN -- Resnet 50000 100 =============================="
echo "Resnet Arch Results:" >> copycat_cifar_results_50000_100_2.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$RESNET" >> copycat_cifar_results_50000_100_2.txt
echo "" >> copycat_cifar_results_50000_100_2.txt

echo "=============================================================="
echo "======================CopycatCNN -- Resnet 50000 80 =============================="
echo "Resnet Arch Results:" >> copycat_cifar_results_50000_80.txt
python3 copycat_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$RESNET" >> copycat_cifar_results_50000_80.txt
echo "" >> copycat_cifar_results_50000_80.txt


