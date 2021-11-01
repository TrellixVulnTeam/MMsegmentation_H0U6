from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import argparse
import os


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Convert VIA dataset into Customdataset for mmsegmentation")
    parser.add_argument("config", 
                        help="config file.")
    parser.add_argument("weight", 
                        help="pretrained weight file in pth form."),
    parser.add_argument("input",
                        help="Specify the input image location."),
    parser.add_argument("--output", default='None',
                        help="Specify the folder location to save segmentation.")

    args = parser.parse_args()
    return args

def segment(config_file, checkpoint_file, file, device='cpu' ):
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device=device)

    # test a single image and show the results
    if args.input:
        img = args.input #'demo\image-1550434545.jpg' #r'demo/Hexa plant 26.10.21.jpg'  # or img = mmcv.imread(img), which will only load it once
    else:
        img = os.path.abspath(file)
    '''TODO: find way to feed images to the inference_segmentor.
    '''
    result = inference_segmentor(model, img)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    model.show_result(img, result, out_file=os.path.join(args.output,os.path.basename(args.input)), opacity=0.5)
    return result


if __name__ == '__main__':

    args = parse_args()

    # config_file = r'work_dirs\fcn_unet_s5-d16_128x128_10k_LeafDataset_T3\fcn_unet_s5-d16_128x128_10k_LeafDataset_T3.py'
    # checkpoint_file = r'work_dirs\fcn_unet_s5-d16_128x128_10k_LeafDataset_T3\iter_10000.pth'
    config_file = args.config
    checkpoint_file = args.weight
    segment(config_file, checkpoint_file)

    

