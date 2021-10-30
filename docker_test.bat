docker pull ccomkhj/mmplant6:v1
docker run -it --rm -w=/Users/samsung_/Hexafarms/Leaf_Area/ -v C:/Users/samsung_/Hexafarms/Leaf_Area/:/work ccomkhj/mmplant6:v1 ^
sh -c "pip install --no-cache-dir -e . ; python3 Hexafarm/predict.py work_dirs\deeplabv3plus_r50-d8_480x480_20k_LeafDataset_T3\deeplabv3plus_r50-d8_480x480_20k_LeafDataset_T3.py work_dirs\deeplabv3plus_r50-d8_480x480_20k_LeafDataset_T3\iter_20000.pth demo\image-1550434545.jpg --output output"

