## streamlitUi One.py


##pip install streamlit

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as NN
from PIL import Image, ImageOps
import torchvision.transforms as transforms


#### load trained model

### if GPU available, use that, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("mnist_model.pth", map_location=device)
model.to(device)

image = image.to(device)  #GPU to CPU Switch