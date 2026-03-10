import struct
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 64
MODEL_PATH = "stegonet_v2.pth"

print(f"Device: {DEVICE}")


class StegoNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.global_avg = nn.AdaptiveAvgPool2d((1,1))
        self.global_max = nn.AdaptiveMaxPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Linear(512*2,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            nn.Sigmoid()
        )


    def forward(self,x):

        x1 = self.pool1(self.conv1(x))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.conv4(x3)

        avg = self.global_avg(x4).flatten(1)
        max_pool = self.global_max(x4).flatten(1)

        combined = torch.cat([avg,max_pool],dim=1)

        return self.fc(combined).squeeze(1)



def load_model(model_path=MODEL_PATH):

    model = StegoNet().to(DEVICE)

    model.load_state_dict(
        torch.load(model_path,map_location=DEVICE)
    )

    model.eval()

    print(f"Model loaded from {model_path}")

    return model



def generate_embedding_map(model,image_path,stride=32,visualize=True):

    model.eval()

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

    H,W = img_rgb.shape[:2]

    print(f"Processing image: {H}x{W}")

    embedding_map = np.zeros((H,W),dtype=np.float32)
    count = np.zeros((H,W),dtype=np.float32)

    transform = transforms.ToTensor()

    positions=[]

    for y in range(0,H-PATCH_SIZE+1,stride):
        for x in range(0,W-PATCH_SIZE+1,stride):
            positions.append((y,x))

    batch_size = 32

    for i in tqdm(range(0,len(positions),batch_size)):

        batch_patches=[]
        batch_positions = positions[i:i+batch_size]

        for y,x in batch_positions:

            patch = img_rgb[y:y+PATCH_SIZE,x:x+PATCH_SIZE].copy()

            patch_tensor = transform(Image.fromarray(patch))

            batch_patches.append(patch_tensor)

        batch_tensor = torch.stack(batch_patches).to(DEVICE)

        with torch.no_grad():
            scores = model(batch_tensor).cpu().numpy()

        for (y,x),score in zip(batch_positions,scores):

            embedding_map[y:y+PATCH_SIZE,x:x+PATCH_SIZE]+=score
            count[y:y+PATCH_SIZE,x:x+PATCH_SIZE]+=1

    mask = count>0
    embedding_map[mask]/=count[mask]

    if visualize:
        plot_embedding_map(img_rgb,embedding_map)

    return embedding_map



def plot_embedding_map(img_rgb,embedding_map):

    fig,axes = plt.subplots(1,3,figsize=(18,6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(embedding_map,cmap="RdYlGn",vmin=0,vmax=1)
    axes[1].set_title("Embedding Priority Map")
    axes[1].axis("off")

    plt.colorbar(im,ax=axes[1])

    overlay = img_rgb.astype(float)/255.0
    heatmap = plt.cm.RdYlGn(embedding_map)[:,:,:3]

    blended = 0.6*overlay + 0.4*heatmap

    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()



def embed_message(image_path,message,embedding_map,output_path,threshold=70):

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    H,W,C = img_rgb.shape

    if embedding_map.shape!=(H,W):
        embedding_map=cv2.resize(embedding_map,(W,H))

    thresh_val = np.percentile(embedding_map,threshold)

    mask = embedding_map>=thresh_val

    y_coords,x_coords = np.where(mask)

    priorities = embedding_map[y_coords,x_coords]

    sorted_idx = np.argsort(priorities)[::-1]

    y_coords = y_coords[sorted_idx]
    x_coords = x_coords[sorted_idx]

    available_bits = len(y_coords)*3
    capacity_bytes = available_bits//8

    msg_bytes = message.encode("utf-8")

    msg_len=len(msg_bytes)

    header = struct.pack(">I",msg_len)

    payload = header + msg_bytes

    payload_bits = "".join(format(b,"08b") for b in payload)

    if len(payload_bits)>available_bits:

        print("Message too large")

        return False,capacity_bytes

    stego = img_rgb.copy()

    bit_idx=0

    for i in range(len(y_coords)):

        if bit_idx>=len(payload_bits):
            break

        y,x = y_coords[i],x_coords[i]

        for ch in range(3):

            if bit_idx>=len(payload_bits):
                break

            val = stego[y,x,ch]

            stego[y,x,ch] = (val & 0xFE) | int(payload_bits[bit_idx])

            bit_idx+=1

    cv2.imwrite(output_path,cv2.cvtColor(stego,cv2.COLOR_RGB2BGR))

    print(f"Saved stego image: {output_path}")

    return True,capacity_bytes



def main_class(image_path,secret_message,threshold=70):

    print("STEGONET ADAPTIVE STEGANOGRAPHY")

    print("Loading model")

    model = load_model(MODEL_PATH)

    print("Generating embedding map")

    embedding_map = generate_embedding_map(model,image_path,stride=32)

    np.save("embedding_map.npy",embedding_map)

    success,capacity = embed_message(
        image_path,
        secret_message,
        embedding_map,
        "stego_output.png",
        threshold
    )

    if not success:
        print("Embedding failed")
        return

    with open("original_message.txt","w",encoding="utf-8") as f:
        f.write(secret_message)

    print("Message saved: original_message.txt")


if __name__ == "__main__":

    MESSAGE = """
Example message used to demonstrate adaptive steganography.
Replace this text with any secret message you want to hide.
"""

    SECRET = 6 * MESSAGE

    start = time.perf_counter()

    main_class(
        image_path="demo/cover_image.png",   # TODO: replace with your cover image
        secret_message=SECRET,
        threshold=70
    )

    end = time.perf_counter()

    print(f"Embedding Time: {end-start:.3f} seconds")