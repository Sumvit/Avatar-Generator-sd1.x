from fastapi.responses import JSONResponse ,FileResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware 
# from fastapi.encoders import jsonable_encoder 
import asyncio
import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np
from fastapi import FastAPI,HTTPException,File,UploadFile,Form
from typing import Optional
from controlnet_aux import OpenposeDetector
import matplotlib.pyplot as plt
import ClipInterrogator as ci
import io
import torch
import boto3
from itertools import product
from diffusers import StableDiffusionControlNetImg2ImgPipeline,ControlNetModel, UniPCMultistepScheduler,LCMScheduler
from diffusers.utils import load_image, make_image_grid

from uuid import uuid4
app=FastAPI()
from botocore.exceptions import NoCredentialsError,ClientError
from dotenv import load_dotenv
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME") 
AWS_REGION= os.getenv("AWS_REGION")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Replace with the URL of your frontend
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
num_steps=15
seed=425657135825768
strengths_guidance = [(0.6,3),(0.7,3.5)]
control_end = [0.5,0.8,1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

controlnets = [
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose"
        ,torch_dtype=torch.float16
    ),
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny"
        ,torch_dtype=torch.float16
    ),
]
model_id_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_id_or_path,
                                                      controlnet=controlnets,
                                                      safety_checker=None,
                                                      requires_safety_checker=False,
                                                      torch_dtype=torch.float16
                                                    )
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5",adapter_name="lcm")
pipe.load_lora_weights("sumvit/sd-1.5-oilCanvas-LoRA",weight_name="oil_canvas3.safetensors",adapter_name="oilCanv")#oil canvas
pipe.load_lora_weights("sumvit/sd-1.5-oilPaintingStyle-LoRA",weight_name="fangxing-000009.safetensors",adapter_name="oilPaint")#oil painting
pipe.load_lora_weights("sumvit/sd-1.5-ghibliStyle-LoRA",weight_name="ghibli_style_offset.safetensors",adapter_name="ghibli")#ghibli style
pipe.load_lora_weights("sumvit/sd-1.5-GTA-Style-LoRA",weight_name="GTA_Style.safetensors",adapter_name="gta")#comic character
upcS=UniPCMultistepScheduler.from_config(pipe.scheduler.config)
lcmS=LCMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = upcS
pipe = pipe.to(device)
adapter_weights=[0,0,0.4,0.8,0.6]
pipe.set_adapters(["lcm","oilCanv","oilPaint","ghibli","gta"],
                adapter_weights=adapter_weights
                )

keywords=['vector art', 'computer graphics', 'by Zvest Apollonio', 'by Zahari Zograf', 'precisionism', 'lyco art', 'a character portrait', 'an ultrafine detailed painting', 'by Patrick Brown', 'by Jan Tengnagel', 'figurative art', 'serial art', 'a character portrait', 'an anime drawing', 'by Ambreen Butt', 'by Zvest Apollonio', 'serial art', 'remodernism', 'a character portrait', 'a comic book panel', 'by Jan Tengnagel', 'by Patrick Brown', 'serial art', 'temporary art', 'a comic book panel', 'a character portrait', 'by Patrick Brown', 'inspired by Patrick Brown', 'plasticien', 'lyco art', 'a character portrait', 'concept art', 'by Lois van Baarle', 'by Lawrence Harris', 'sots art', 'serial art', 'a character portrait', 'concept art', 'by Lois van Baarle', 'by Paul Kelpe', 'sots art', 'furry art', 'a character portrait', 'a portrait', 'by Patrick Brown', 'by Lois van Baarle', 'fantastic realism', 'serial art', 'a character portrait', 'an ultrafine detailed painting', 'by John Steell', 'by Paul Kelpe', 'serial art', 'new objectivity', 'a character portrait', 'a comic book panel', 'by Patrick Brown', 'by Steve Dillon', 'abstract illusionism', 'altermodern', 'a character portrait', 'concept art', 'by Patrick Brown', 'by Lawrence Harris', 'serial art', 'sots art']
triggers=["oil canvas","oil painting","ghibli style","comic character"]
t=",".join([ x for x,y in zip(triggers,adapter_weights) if y>0])
k=",".join(keywords)
def upload_to_s3(file_path, bucket_name, s3_file_name):
    session =boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    s3 = session.client('s3')

    try:
        s3.upload_file(file_path, bucket_name, s3_file_name)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="The file was not found.")
    except ClientError as e:
        raise HTTPException(status_code=500, detail="Error uploading file to S3.")

def get_s3_file_url(bucket_name, region, s3_file_name):
    return f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_file_name}"
async def get_canny_image(image:Image.Image):
    image=image
    image=np.array(image)
    low_threshold =50
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold,apertureSize=3,L2gradient=True)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image=Image.fromarray(image)
    return canny_image
async def  get_openpose_image(image: Image.Image):
    original_image=image
    openpose_image = openpose(original_image,include_hand=False, include_face=True,include_body=True)
    return openpose_image
async def  get_clip_interrogator_output(image:Image.Image):
    return ci.image_to_prompt(image,"fast")

async def reset():
    num_steps=15
    seed=425657135825768
    strengths_guidance = [(0.6,3),(0.7,3.5)]
    control_end = [0.5,0.8,1]
    adapter_weights=[0,0,0.4,0.8,0.6]
    pipe.scheduler = upcS
def control_net(init_image,canny_image,openpose_image,CLIP_int=""):

  prompt = f"vivid colors,vector graphic,2d line art,outlined features,clearly drawn, well defined shapes,sharp drawing,{t},{CLIP_int},{k}"
  negative_prompt="side eye,cross eye,angry,disformed face,disfigured lips, disfigured nose,poorly Rendered face,poorly drawn face,poor facial details,poorly drawn hands,poorly rendered hands,low resolution,Images cut out at the top, left, right, bottom.,bad composition,mutated body parts,blurry image,disfigured,oversaturated,bad anatomy,deformed body features,NSFW,nudity"
  input_ids = pipe.tokenizer(
      prompt,
      return_tensors="pt",
      padding="max_length",
      max_length=77,
      truncation=False
  ).input_ids.to(device)
  max_length=77

  negative_ids = pipe.tokenizer(
      negative_prompt,
      truncation=False,
      padding="max_length",
      max_length=input_ids.shape[-1],
      return_tensors="pt"
  ).input_ids.to(device)
  concat_embeds = []
  neg_embeds = []
  for i in range(0, input_ids.shape[-1], max_length):
      concat_embeds.append(
          pipe.text_encoder(
              input_ids[:, i: i + max_length]
          )[0]
      )
      neg_embeds.append(
          pipe.text_encoder(
              negative_ids[:, i: i + max_length]
          )[0]
      )

  prompt_embeds = torch.cat(concat_embeds, dim=1)
  negative_prompt_embeds = torch.cat(neg_embeds, dim=1) 
  images=[]
  for (strength, guidance_scale), end  in product(strengths_guidance, control_end):
    images.append(pipe(
        
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    control_image=[openpose_image,canny_image],
                   
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    control_guidance_start=0 ,
                    control_guidance_end=end,
                    controlnet_conditioning_scale=[0.5,0.4]#@param
        ,
                    generator=torch.manual_seed(seed),
                    num_inference_steps=num_steps).images[0])
  return images
@app.get("/")
def read_root():
    return {
        "message": "Avatar Generator API",
        "usage": "Send a POST request to the /avatar_generator/ endpoint with an image file",
        "example1": {
            "endpoint": "/avatar_generator/",
            "method": "POST",
            "file_type": "image",
        },
        "example2":{
              "endpoint": "/upscale/",
            "method": "POST",
            "file_type": "image",
        }
}
@app.post("/avatar_generator/")
async def upload_file(image_file: UploadFile=File(...),LCM: bool=Form(default=False)):
  try:
    content=await image_file.read()  #UploadFile.read()
    photo = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) 
    generated_path=os.path.join(os.getcwd(),str(UPLOAD_DIR) ,"generated_image.jpg")
    image=Image.fromarray(np.array(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB) ))
    unique_id = str(uuid4())
    x,y=image.size 
    if x==y:
        image=image.resize((512,512))
        if LCM:
            num_steps=5
            seed=425657135825768
            strengths_guidance = [(0.6,1.3),(0.65,1.5),(0.7,2)]
            control_end = [0.5,0.8]
            adapter_weights=[1,0,0.4,0.8,0.6]
            pipe.scheduler = lcmS
        canny_image, openpose_image, description = await asyncio.gather(
            get_canny_image(image),
            get_openpose_image(image),
            get_clip_interrogator_output(image)
        )
    
        try:    
            generated_images=control_net(init_image=image,canny_image=canny_image,openpose_image=openpose_image,CLIP_int=description)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error while Generating image")
        links=[]
        for i in range(len(generated_images)):
            filename=generated_path.split(".jpg")[0]+f"{i}"+".jpg"
            generated_images[i].save(filename)
            upload_to_s3(filename, S3_BUCKET_NAME, f'Avatars/{unique_id}/{os.path.basename(filename)}')
            links.append(get_s3_file_url(S3_BUCKET_NAME, AWS_REGION, f'Avatars/{unique_id}/{os.path.basename(filename)}'))
            os.remove(filename)
        LCM and await reset()
        return JSONResponse({
            "unique_id":unique_id,
            "avatars":links
        })
        # return FileResponse(generated_path,media_type="image/jpg")
    else:
        raise HTTPException(status_code=400,detail="Image aspect ratio must be 1:1, try again. ")
    #return 
  except Exception as e:
        # print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error uploading image. Please try again.")


from RealESRGAN import RealESRGAN

model = RealESRGAN(device, scale=2)
model.load_weights(f'weights/RealESRGAN_x{2}.pth', download=True)
model2 = RealESRGAN(device, scale=4)
model2.load_weights(f'weights/RealESRGAN_x{4}.pth', download=True)
model3 = RealESRGAN(device, scale=8)
model3.load_weights(f'weights/RealESRGAN_x{8}.pth', download=True)
# model.load_weights(f'weights/RealESRGAN_x{8}.pth', download=True)
@app.post("/upscale/")
async def upscale_image(image_file: UploadFile=File(...),scale:int=Form(default=4),unique_id:Optional[str]=Form(default=None)):
    try:
        if scale in [2,4,8]:
            content=await image_file.read()  #UploadFile.read()
            photo = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            image=Image.fromarray(np.array(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB) ))
            sr_image=None
            if scale==2 :
                sr_image= model.predict(image) 
            elif scale==4:
                sr_image= model2.predict(image)
            elif scale==8 :
                sr_image= model3.predict(image)
            # if(scale==8):
            #     sr_image=sr_image.resize((1024,1024))
            #     sr_image = model.predict(sr_image)
            sr_image=sr_image.resize((512,512))
            filename=os.path.join(os.getcwd(),str(UPLOAD_DIR) ,"upscaled_image.jpg")
            sr_image.save(filename)
            if unique_id==None:
                unique_id=str(uuid4())
            upload_to_s3(filename, S3_BUCKET_NAME, f'Avatars/{unique_id}/{os.path.basename(filename)}')
            link=get_s3_file_url(S3_BUCKET_NAME, AWS_REGION, f'Avatars/{unique_id}/{os.path.basename(filename)}')
            os.remove(filename)
            return JSONResponse({
                "unique_id":unique_id,
                "upscaled_image":link
            })
        else:
            raise HTTPException(status_code=400,detail="scale needs to be either 4 or 8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error uploading image. Please try again.")