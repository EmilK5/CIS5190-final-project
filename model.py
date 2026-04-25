import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from typing import Any, Iterable, List

# --- YOUR CUSTOM ARCHITECTURE BLOCKS ---
class ResidualRegressionBlock(nn.Module):
    def __init__(self, size, dropout_p=0.2):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        residual = x
        out = self.dropout(self.gelu(self.bn(self.fc(x))))
        return out + residual

class EfficientNetGPS(nn.Module):
    def __init__(self, dropout_p=0.3):
        super(EfficientNetGPS, self).__init__()
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            ResidualRegressionBlock(512, 0.2), 
            nn.Linear(512, 128),               
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 2)                  
        )

    def forward(self, x):
        if self.training:
            return self.backbone(x)
        else:
            # GPS-Safe Invisible TTA (Zoom & Brightness)
            out_orig = self.backbone(x)
            x_zoom = torch.nn.functional.interpolate(
                x[:, :, 16:-16, 16:-16], size=(448, 448), mode='bilinear'
            )
            out_zoom = self.backbone(x_zoom)
            x_bright = torch.clamp(x * 1.15, min=-3.0, max=3.0) 
            out_bright = self.backbone(x_bright)
            return (out_orig + out_zoom + out_bright) / 3


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model = EfficientNetGPS(dropout_p=0.3)
        self.register_buffer('origin', torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor([1.0, 1.0], dtype=torch.float32))

    def forward(self, x):
        local_meters = self.base_model(x)
        dx_easting = local_meters[:, 0]
        dy_northing = local_meters[:, 1]
        
        lat_offset = dy_northing / self.scale[0]
        lon_offset = dx_easting / self.scale[1]
        
        pred_lat = self.origin[0] + lat_offset
        pred_lon = self.origin[1] + lon_offset
        return torch.stack([pred_lat, pred_lon], dim=1)

    def eval(self) -> None:
        super().eval()

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        device = next(self.parameters()).device
        self.eval() # Ensure dropout is off and TTA is active
        
        # Handle batched tensors or lists of tensors from the grader
        if isinstance(batch, torch.Tensor):
            inputs = batch.to(device)
        else:
            inputs = torch.stack(list(batch)).to(device)
            
        with torch.no_grad():
            preds = self.forward(inputs)
            
        return preds.cpu().tolist() # Returns [[lat1, lon1], [lat2, lon2], ...]

def get_model() -> Model:
    return Model()