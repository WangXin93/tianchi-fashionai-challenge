#!/usr/bin/env sh

#python3 bin/transfer_learning.py --attribute coat_length_labels --epochs 50 --model resnet18 --save_folder resnet18-larger; \
#python3 bin/transfer_learning.py --attribute collar_design_labels --epochs 50 --model inception_v3 --save_folder inception_v3-larger; \
#python3 bin/transfer_learning.py --attribute lapel_design_labels --epochs 50 --model inception_v3 --save_folder inception_v3-larger; \
#python3 bin/transfer_learning.py --attribute neck_design_labels --epochs 50 --model inception_v3 --save_folder inception_v3-larger; \
#python3 bin/transfer_learning.py --attribute neckline_design_labels --epochs 50 --model inception_v3 --save_folder inception_v3-larger; \
#python3 bin/transfer_learning.py --attribute pant_length_labels --epochs 50 --model resnet18 --save_folder resnet18-larger; \
#python3 bin/transfer_learning.py --attribute skirt_length_labels --epochs 50 --model resnet18 --save_folder resnet18-larger; \
#python3 bin/transfer_learning.py --attribute sleeve_length_labels --epochs 50 --model resnet18 --save_folder resnet18-larger;
#

python3 bin/transfer_learning.py --attribute coat_length_labels --epochs 50 --model inception_v3 --save_folder spam --img_size 299; \
python3 bin/transfer_learning.py --attribute sleeve_length_labels --epochs 50 --model inception_v3 --save_folder spam --img_size 299; \
python3 bin/transfer_learning.py --attribute pant_length_labels --epochs 50 --model inception_v3 --save_folder spam --img_size 299
