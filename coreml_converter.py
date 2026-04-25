import torch
import coremltools as ct
import numpy as np

from model import get_model


# Load PyTorch model architecture
model = get_model()

# Load trained weights
checkpoint = torch.load("model.pt", map_location="cpu")

if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
    print("Saved checkpoint is a state_dict, loading weights into model...")
    model.load_state_dict(checkpoint)
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    print("Saved checkpoint contains 'state_dict', loading weights into model...")
    model.load_state_dict(checkpoint["state_dict"])
else:
    print("Saved checkpoint is a full model, using it directly...")
    model = checkpoint

# Set model to evaluation mode
model.eval()


# Create dummy input matching your preprocessing:
# batch size 1, 3 color channels, 448x448 image
example_input = torch.randn(1, 3, 448, 448)


# Trace the model
traced_model = torch.jit.trace(model, example_input)


# 5. Convert to Core ML
# PyTorch preprocessing is:
# image / 255, then (x - mean) / std
#
# Core ML image input uses:
# normalized = pixel * scale + bias
#
# So for each channel:
# scale = 1 / (255 * std)
# bias = -mean / std

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

scale = 1.0 / (255.0 * std)
bias = -mean / std

mlmodel = ct.convert(
    traced_model,
    source="pytorch",
    inputs=[
        ct.ImageType(
            name="image",
            shape=example_input.shape,
            color_layout=ct.colorlayout.RGB,
            scale=scale,
            bias=bias,
        )
    ],
    outputs=[
        ct.TensorType(name="coordinates")
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
)


# Save as .mlpackage
mlmodel.save("EfficientNetGPS.mlpackage")