# Preprocess the dataset and train/convert the VWW model.
#python3 parse_coco.py annotations/instances_train2017.json
python3 train_vww.py
python3 convert_vww.py trained_models/vww_96.h5