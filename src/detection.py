import os
from copy import deepcopy

import torch

# モデル読み込み
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 画像
#imgs = ['https://ultralytics.com/images/zidane.jpg']  # <- 自由に変更してね！
imgs = "image/detection/input.jpg" # <- 自由に変更してね！
img_paths = deepcopy(imgs)

# 推論
results = model(imgs)

# 可視化
save_dir = "image/detection"
os.makedirs(save_dir, exist_ok=True)
results.save(save_dir) 
xys = results.pandas().xyxy
for xy, img in zip(xys, img_paths):
    basename = os.path.splitext(os.path.basename(img))[0]
    xy.to_csv(f"{save_dir}/{basename}.csv")
    print(xy)
