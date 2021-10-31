docker pull ccomkhj/mmplant6:v4
docker run -it --rm -w=/work -v C:/Users/samsung_/Hexafarms/Leaf_Area/mmsegmentation/:/work ccomkhj/mmplant6:v4 ^
sh -c "pip install --no-cache-dir -e . ; python3 predict.py z_input/fcn_unet_s5-d16_128x128_10k_LeafDataset_T3.py z_input/fcn_unet_iter_10000.pth demo/Hexa-plant261021.jpg --output output"

