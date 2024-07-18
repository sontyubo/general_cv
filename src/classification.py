import os
from PIL import Image
import urllib.request

import matplotlib
matplotlib.use('Agg')  # コンテナでアクセスできないGUIを使用しない
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

#from utils.imagenet_labels import idx2label 

# モデル読み込み
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model.eval()

# 画像をurlからダウンロードして、指定したディレクトリに保存する
url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg" # <- 自由に変更してね
save_dir = "image/classification"
os.makedirs(save_dir, exist_ok=True)
filename = f"{save_dir}/{os.path.basename(url)}"
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
input_image = Image.open(filename)

#前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) 

#推論
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)

#可視化
idx = torch.nn.functional.softmax(output[0], dim=0).argmax().item()
# 判定したタグを出力する
#print(idx2label[idx])
plt.imshow(input_image)
#plt.title(idx2label[idx])
plt.savefig(filename)

