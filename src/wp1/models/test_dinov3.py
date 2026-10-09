import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


if __name__ == "__main__": 
    pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="cuda:0", 
    )
    image = Image.open('/home/labsig/Documents/Axelle/Main research/Data/uliege/train/camelyon16_1/27666492_69888_123648_768_768.png').convert('RGB')

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
       
    pooled_output = outputs.pooler_output
    print("Pooled output shape:", pooled_output.shape)


