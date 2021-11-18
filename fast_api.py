from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from predict import segment_api as segment

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hexa": "Farm"}

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

    segment(config_file, checkpoint_file, input_dir, output_dir) 
    image = open(output_dir+f"/{file.filename}", 'rb')

    return StreamingResponse(image, media_type=("image/jpeg"or"image/png"))
    