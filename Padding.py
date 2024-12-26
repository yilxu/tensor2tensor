import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# 1) Read training dataset Excel file
df = pd.read_excel("/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/training_data.xlsx")

# 2) Create a folder structure for label
#    For demonstration, everything goes under label "0"
os.makedirs("myDataForMNIST/0", exist_ok=True)

# 3) Convert each row to a 28x28 image
target_size = 784  

for i, row in df.iterrows():
    # Convert row to a NumPy array
    arr = row.values  # shape: (num_features,)
    
    # If fewer than 784 features, pad with zeros
    if arr.shape[0] < target_size:
        diff = target_size - arr.shape[0]
        arr = np.pad(arr, (0, diff), mode='constant', constant_values=0)
    else:
        # If there are more than 784, just truncate to first 784
        arr = arr[:target_size]
    
    # Reshape to (28, 28)
    arr_2d = arr.reshape(28, 28)
    
    # scale values to 0–255 (typical 8-bit grayscale)
    # Here we assume values are in [0,1]
    arr_2d_scaled = (arr_2d * 255).astype(np.uint8)
    
    # Save the image as PNG into "myDataForMNIST/0/"
    out_path = f"myDataForMNIST/0/row_{i}.png"
    Image.fromarray(arr_2d_scaled).save(out_path)

# 4) Print the last image processed image to see how it looks
plt.imshow(arr_2d_scaled, cmap='gray')
plt.title("Example: Last Row Processed as a 28x28 Image")
plt.show()

# Visualize one sample from the .ubyte file
# Load & Inspect the Data
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Point to the folder with new ubyte files.
# The standard MNIST dataset class in torchvision expects the files to be in a
# "MNIST/raw" subfolder or the same structure. 
dataset = datasets.MNIST(
    root='.',         
    train=True,
    download=False,    
    transform=transforms.ToTensor()
)

# Now pick a sample to visualize
img, label = dataset[0]  # The first sample
print("Label:", label)
print("Image shape:", img.shape)  # Typically [1, 28, 28] in PyTorch

plt.imshow(img.squeeze(0), cmap='gray')
plt.title(f"Label = {label}")
plt.show()

