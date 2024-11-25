#!/bin/bash
DATASET_ID=vww
DATASET_SIZE=1000
NUM_CLASSES=2
BASIC=basic
VGG=vgg
LENET=lenet
ALEXNET=alexnet
RESNET=resnet
MOB=mobilenet
NB_EPOCS_100=100
NB_EPOCS_80=80
NB_STOLEN=50000

QNN_INT8='models/vww_96_testable_logits_quant.tflite'

echo "=============================================================="
echo "======================KnockoffNets -- Basic 50000 100 =============================="
echo "Basic Arch Results:" >> knockoff_vww_results_50000_100_2.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$BASIC" >> knockoff_vww_results_50000_100_2.txt
echo "" >> knockoff_vww_results_50000_100_2.txt

echo "=============================================================="
echo "======================KnockoffNets -- Basic 50000 80 =============================="
echo "Basic Arch Results:" >> knockoff_vww_results_50000_80.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$BASIC" >> knockoff_vww_results_50000_80.txt
echo "" >> knockoff_vww_results_50000_80.txt

echo "=============================================================="
echo "======================KnockoffNets -- lenet 50000 100 =============================="
echo "Lenet Arch Results:" >> knockoff_vww_results_50000_100_2.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$LENET" >> knockoff_vww_results_50000_100_2.txt
echo "" >> knockoff_vww_results_50000_100_2.txt

echo "=============================================================="
echo "======================KnockoffNets -- lenet 50000 80 =============================="
echo "Lenet Arch Results:" >> knockoff_vww_results_50000_80.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$LENET" >> knockoff_vww_results_50000_80.txt
echo "" >> knockoff_vww_results_50000_80.txt

echo "=============================================================="
echo "======================KnockoffNets -- Alexnet 50000 100 =============================="
echo "Alexnet Arch Results:" >> knockoff_vww_results_50000_100_2.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$ALEXNET" >> knockoff_vww_results_50000_100_2.txt
echo "" >> knockoff_vww_results_50000_100_2.txt

echo "=============================================================="
echo "======================KnockoffNets -- Alexnet 50000 80 =============================="
echo "Alexnet Arch Results:" >> knockoff_vww_results_50000_80.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$ALEXNET" >> knockoff_vww_results_50000_80.txt
echo "" >> knockoff_vww_results_50000_80.txt

echo "=============================================================="
echo "======================KnockoffNets -- Resnet 50000 100 =============================="
echo "Resnet Arch Results:" >> knockoff_vww_results_50000_100_2.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$RESNET" >> knockoff_vww_results_50000_100_2.txt
echo "" >> knockoff_vww_results_50000_100_2.txt

echo "=============================================================="
echo "======================KnockoffNets -- Resnet 50000 80 =============================="
echo "Resnet Arch Results:" >> knockoff_vww_results_50000_80.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$RESNET" >> knockoff_vww_results_50000_80.txt
echo "" >> knockoff_vww_results_50000_80.txt

echo "=============================================================="
echo "======================KnockoffNets -- Mobile 50000 100 =============================="
echo "Mobilenet Arch Results:" >> knockoff_vww_results_50000_100_2.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_100" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$MOB" >> knockoff_vww_results_50000_100_2.txt
echo "" >> knockoff_vww_results_50000_100_2.txt

echo "=============================================================="
echo "======================KnockoffNets -- Mobile 50000 80 =============================="
echo "Mobilenet Arch Results:" >> knockoff_vww_results_50000_80.txt
python3 knockoff_nets_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --nb_epochs "$NB_EPOCS_80" --nb_stolen "$NB_STOLEN" --target_int8 "$QNN_INT8" --theived_template "$MOB" >> knockoff_vww_results_50000_80.txt
echo "" >> knockoff_vww_results_50000_80.txt