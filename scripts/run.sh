#!/usr/bin/env sh

# coat_length_labels 
# collar_design_labels 
# lapel_design_labels 
# neck_design_labels 
# neckline_design_labels 
# pant_length_labels 
# skirt_length_labels 
# sleeve_length_labels 


python3 bin/transfer_learning.py --model resnet18 --attribute skirt_length_labels --epochs 43 --save_folder resnet18_a --img_size 224 --batch_size 32 --csv_folder fashionAI_a; \

python3 bin/transfer_learning.py --model resnet18 --attribute skirt_length_labels --epochs 43 --save_folder resnet18_b --img_size 224 --batch_size 32 --csv_folder fashionAI_b; \

python3 bin/transfer_learning.py --model resnet18 --attribute skirt_length_labels --epochs 43 --save_folder resnet18_c --img_size 224 --batch_size 32 --csv_folder fashionAI_c; \

python3 bin/transfer_learning.py --model resnet18 --attribute skirt_length_labels --epochs 43 --save_folder resnet18_d --img_size 224 --batch_size 32 --csv_folder fashionAI_d
