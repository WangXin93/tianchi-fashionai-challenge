#!/usr/bin/env sh

python3 bin/transfer_learning.py --attribute coat_length_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False; \
python3 bin/transfer_learning.py --attribute collar_design_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False; \
python3 bin/transfer_learning.py --attribute lapel_design_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False; \
python3 bin/transfer_learning.py --attribute neck_design_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False; \
python3 bin/transfer_learning.py --attribute neckline_design_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False; \
python3 bin/transfer_learning.py --attribute pant_length_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False; \
python3 bin/transfer_learning.py --attribute skirt_length_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False; \
python3 bin/transfer_learning.py --attribute sleeve_length_labels --model resnet101 --epochs 100 --save_folder resnet101-zero --pretrained False;
