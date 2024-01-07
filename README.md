## Finetuning diffusion model with LoRA

### 一、 File Description
- `datasets`: This directory is used to store datasets, including `DonaldTrump80`、 `lyf30-512-512` 、`sjl50-512-512`.
- `ddpm`: Code for Denoising Diffusion Probabilistic Model
- `dreambooth`: Code for Dreambooth Model
- `outputs`: This directory is used to store the output images of various models.
- `scripts`: This directory is used to store training or generate scripts.
- `text2img`: Code for Text2img Model Based on SD1.5.
- `env_linux.yml`: Installation environment on Linux platform.
- `env_windows.yml`: Installation environment on Windows platform.

### 二、Preparation

#### 2.1 Create Environment
```bash
conda env create -f env_linux.yml
or
conda env create -f env_windows.yml
```

#### 2.2 Download DDPM Model (Optional)
```bash
python ddpm/download.py
```

#### 2.3 Download SD1.5
```bash
python ddpm/download.py
```

#### 2.4 Download LoRA Finetuning Model
```bash
python text2img/lora-finetune/lyf/download.py
python text2img/lora-finetune/Scarlett/download.py
```

### 三、Train and Generate
#### 3.1 Train Text2img Model
```bash
bash scripts/train_text2img.sh
```
#### 3.2 Generate Images
```bash
bash scripts/generate_text2img.sh
```

### 四、Display

#### 4.1 DT

<div><center>
<img src=outputs/text2img/DT/td-04.png width=70% height=70% >
<br>
</center></div>

#### 4.2 LYF
<div><center>
<img src=outputs/text2img/lyf/lyf-02.png width=70% height=70% >
<br>
</center></div>

#### 4.3 Scarlett
<div><center>
<img src=outputs/text2img/Scarlett/sjl-05.png width=70% height=70% >
<br>
</center></div>

