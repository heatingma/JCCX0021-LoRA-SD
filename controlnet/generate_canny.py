import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import numpy as np


dir = 'training/tlp/train'

for file in tqdm(os.listdir(os.path.join(dir, 'original'))):
    path = os.path.join(dir, 'original', file)

    ori_img = cv2.imread(path)

    resized_img = cv2.resize(ori_img, (512, 512))

    edges = cv2.Canny(resized_img, threshold1=150, threshold2=300)

    idx = file[:-4]

    os.makedirs(os.path.join(dir, 'source'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'target'), exist_ok=True)
    cv2.imwrite(os.path.join(dir, 'source', file), edges)
    cv2.imwrite(os.path.join(dir, 'target', file), resized_img)

data = []
with open(os.path.join(dir, 'metadata.jsonl'), 'r') as f:
    for line in f:
        line = json.loads(line.strip())
        item = {"source": f"source/{line['file_name']}", "target": f"target/{line['file_name']}", "prompt": line["additional_feature"]}
        data.append(item)

with open(os.path.join(dir, 'prompt.json'), 'w') as f:
    for item in data:
        json_line = json.dumps(item)
        f.write(json_line + '\n')

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        self.dir = dir
        with open(os.path.join(self.dir, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.dir, source_filename))
        target = cv2.imread(os.path.join(self.dir, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


# dataset = MyDataset()
# print(len(dataset))
#
# item = dataset[0]
# jpg = item['jpg']
# txt = item['txt']
# hint = item['hint']
# print(txt)
# print(jpg.shape)
# print(hint.shape)