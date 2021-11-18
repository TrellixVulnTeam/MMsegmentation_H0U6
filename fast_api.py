from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
import torch

def segment(config_file, checkpoint_file, input_dir, output_dir, device='cuda:0' ):
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device=device)
    img = mmcv.imread(input_dir)
    result = inference_segmentor(model, img)
    model.show_result(img, result, out_file=os.path.join(output_dir, os.path.basename(input_dir)), opacity=0.5)
    return result

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/segment")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"fast_api/input/{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    ############################ Current Best Model ###############################
    config_file = "fast_api/best_model/fcn_unet_s5-d16_128x128_10k_LeafDataset_T3.py"
    checkpoint_file = "fast_api/best_model/iter_10000.pth"
    ################################################################################

    input_dir = f"fast_api/input/{file.filename}"
    output_dir = "fast_api/output/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    segment(config_file, checkpoint_file, input_dir, output_dir, device) 
    image = open(output_dir+f"/{file.filename}", 'rb')

    return StreamingResponse(image, media_type=("image/jpeg"or"image/png"))
    