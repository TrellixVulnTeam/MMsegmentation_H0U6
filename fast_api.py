from fastapi.datastructures import UploadFile
from starlette.responses import FileResponse
from torch.cuda import check_error
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
    parser.add_argument("--save",
                        action='store_true',
                        help='whether to save output result files')


    args = parser.parse_args()
    return args

def segment(config_file, checkpoint_file, input_dir, output_dir, device='cuda:0' ):
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device=device)
    img = mmcv.imread(input_dir)
 
    result = inference_segmentor(model, img)

    model.show_result(img, result, out_file=os.path.join(output_dir, os.path.basename(input_dir)), opacity=0.5)
    return result

def segment_api(config_file, checkpoint_file, img):
    model = init_segmentor(config_file, checkpoint_file, device='cpu')
    result = inference_segmentor(model, img)
    return result
    

from fastapi.responses import FileResponse, StreamingResponse
from fastapi import FastAPI, UploadFile, File
import numpy as np
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/segment")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"fastapi-files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())


    config_file = "../z_input/fcn_unet_s5-d16_128x128_10k_LeafDataset_T3.py"
    checkpoint_file = "../z_input/iter_10000.pth"
    input_dir = f"fastapi-files/{file.filename}"
    output_dir = "fastapi-files/"

    segment(config_file, checkpoint_file, input_dir, output_dir) 
    image = open(f"fastapi-files/{file.filename}", 'rb')

    return StreamingResponse(image, media_type="image/jpeg")
    