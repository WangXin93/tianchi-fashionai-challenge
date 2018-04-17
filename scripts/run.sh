#!/usr/bin/env sh

# coat_length_labels 
# collar_design_labels 
# lapel_design_labels 
# neck_design_labels 
# neckline_design_labels 
# pant_length_labels 
# skirt_length_labels 
# sleeve_length_labels 


#python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute neck_design_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
#python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute coat_length_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute collar_design_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute lapel_design_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute neckline_design_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute pant_length_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute skirt_length_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
python3 bin/transfer_learning.py --model inceptionresnetv2 --attribute sleeve_length_labels --epochs 50 --save_folder inceptionresnetv2 --img_size 299 --batch_size 8 --verbose; \
