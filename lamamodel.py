import numpy as np
from lama_cleaner.parse_args import parse_args
import random
import time
import imghdr
from typing import Union

import cv2
import torch
from loguru import logger
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler


from enum import Enum

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model: ModelManager(name="lama", device=device)
device = torch.device(device)
input_image_path: str = None
is_disable_model_switch: bool = False
is_desktop: bool = False
from lama_cleaner.helper import (
    resize_max_size,
)


def diffuser_callback(i, t, latents):
    pass
def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w
class LDMSampler(str, Enum):
    ddim = "ddim"
    plms = "plms"
def get_data(img_p ,mask_p):
    img = cv2.imread(str(img_p))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    mask = cv2.dilate(
        mask,
        np.ones((10, 10), np.uint8),
        iterations=1
    )
    img = cv2.resize(img, None, fx=1, fy= 1.0, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=1, fy= 1.0, interpolation=cv2.INTER_NEAREST)
    return img, mask

def process(img_p, mask_p):
    image, mask = get_data(img_p=img_p, mask_p=mask_p)
    alpha_channel = image[:, :, -1]
    if image.shape[:2] != mask.shape[:2]:
        return f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}", 400

    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    size_limit: Union[int, str] = 2500

    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    config = Config(
        ldm_steps=1,
        ldm_sampler=LDMSampler.plms,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200,
    )
    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)

    logger.info(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    logger.info(f"Resized image shape: {image.shape}")

    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    start = time.time()

    res_np_img = model(image, mask, config)  # -----------------------导入模型

    return res_np_img


def lamamain(image,mask,name):
    args = parse_args()
    global model
    global device
    global input_image_path
    global is_disable_model_switch
    global is_desktop

    device = torch.device(args.device)
    input_image_path = args.input
    is_disable_model_switch = args.disable_model_switch
    is_desktop = args.gui
    if is_disable_model_switch:
        logger.info(f"Start with --disable-model-switch, model switch on frontend is disable")

    model = ModelManager(
        name=name,
        device=device,
        hf_access_token=args.hf_access_token,
        sd_disable_nsfw=args.sd_disable_nsfw,
        sd_cpu_textencoder=args.sd_cpu_textencoder,
        sd_run_local=args.sd_run_local,
        callback=diffuser_callback,
    )
    image=process(image,mask)
    image=np.uint8(image)
    images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return images
    # cv2.imwrite('imagemask.jpg', images)
    # print(images)
if __name__ == '__main__':
    lamamain()