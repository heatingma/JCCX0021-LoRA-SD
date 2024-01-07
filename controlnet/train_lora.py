import json
import os.path

import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from lora_diffusion import inject_trainable_lora
import loralib as lora
import torch
from torch import nn


def inject_lora(module, name=None, ancestor=None):
    children = list(module.named_children())
    if len(children) > 0:
        for name, sub_module in children:
            inject_lora(sub_module, name, module)
    else:
        # if isinstance(module, nn.Conv2d):
        #     old_weight = getattr(ancestor, name).weight
        #     old_bias = getattr(ancestor, name).bias
        #     old_stride = getattr(ancestor, name).stride
        #     old_padding = getattr(ancestor, name).padding
        #     setattr(ancestor, name, lora.Conv2d(
        #     module.in_channels,
        #     module.out_channels,
        #     kernel_size=module.kernel_size[0],
        #     r=1,
        #     lora_alpha=2))
        #     getattr(ancestor, name).conv.weight = old_weight
        #     getattr(ancestor, name).conv.bias = old_bias
        #     getattr(ancestor, name).conv.padding = old_padding
        #     getattr(ancestor, name).conv.stride = old_stride

        # elif isinstance(module, nn.Linear):
        if isinstance(module, nn.Linear):
            weight = getattr(ancestor, name).weight
            setattr(ancestor, name, lora.Linear(
            module.in_features,
            module.out_features,
            r=8,
            lora_alpha=2,
            initial_weight=weight))


class MyDataset(Dataset):
    def __init__(self, dir='./training/lyf', training=True):
        self.data = []
        if training:
            self.dir = os.path.join(dir, 'train')
        else:
            self.dir = os.path.join(dir, 'test')
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

if __name__ == '__main__':
    # Configs
    control_path = './models/control_sd15_canny.pth'
    batch_size = 2
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    state_dict = torch.load(control_path)
    # del state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml')
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    print(model.load_state_dict(state_dict))

    # inject_lora(model.control_model)

    # Misc
    dataset = MyDataset()

    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs=100)

    # Train!
    trainer.fit(model, dataloader)

    # net = nn.Sequential(nn.Conv2d(2, 2, 2))
    # inject_lora(net)
    # print(net)