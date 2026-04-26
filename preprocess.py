import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from typing import Tuple

def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 1. Inference transform
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. Extract the folder path where the grader's CSV is located
    csv_directory = os.path.dirname(path)
    
    df = pd.read_csv(path)
    X = []
    y = []
    
    # 3. Iterate through the CSV
    for _, row in df.iterrows():
        img_path = os.path.join(csv_directory, row['file_name'])
        
        # Open the image
        image = Image.open(img_path).convert("RGB")
        X.append(inference_transform(image))
        
        y.append([row['Latitude'], row['Longitude']])
        
    # 4. Stack inputs
    X_tensor = torch.stack(X)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor