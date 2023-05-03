from metaseg import SahiAutoSegmentation,sahi_sliced_predict
import cv2
from tqdm import tqdm
import os
from lamamodel import lamamain
import numpy as np

def load_video(video_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    return cap, out
def lama_image_app(
    image_path,
    sam_model_type,
    selected_lamamodel,
    detection_model_type,
    detection_model_path,
    conf_th,
    image_size,
    slice_height,
    slice_width,
    overlap_height_ratio,
    overlap_width_ratio,
):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    width = img.shape[1]
    height = img.shape[0]
    boxes = sahi_sliced_predict(
        image_path=img,
        detection_model_type=detection_model_type,  # yolov8, detectron2, mmdetection, torchvision
        detection_model_path=detection_model_path,
        conf_th=conf_th,
        image_size=image_size,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    if len(boxes) == 0:
        boxes = [0, 0, 0, 0]

    masks =SahiAutoSegmentation().predict(
        source=img,
        model_type=sam_model_type,
        input_box=boxes,
        multimask_output=False,
        random_color=False,
        show=False,
    )
    a = np.full((1, height, width), False)
    for idx, mask in enumerate(masks):
        if mask[0] != None:
            bmasks = mask.detach().cpu().numpy()
            mask2 = np.logical_or(a, bmasks)
            a[:, :, :] = mask2
    h, w = a.shape[-2:]
    mask_image = a.reshape(h, w, 1)
    cv2.imwrite('./output/aimage/image.jpg', img)
    cv2.imwrite("output/amask/mask.jpg" , mask_image * 255)
    mask_image = mask_image * img
    tmp = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(mask_image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)


    cv2.imwrite("output/aimages/images.png" , dst)

    lamaindex = lamamain("output/aimage/image.jpg", "output/amask/mask.jpg" , selected_lamamodel)
    lamaimage = cv2.cvtColor(lamaindex, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output/alama/imagemask.jpg' , lamaimage)

    lists=["output/aimage/image.jpg","output/amask/mask.jpg","output/aimages/images.png","output/alama/imagemask.jpg"]


    return lists



def lama_video_app(
    image_path,
    sam_model_type,
    selected_lamamodel,
    detection_model_type,
    detection_model_path,
    conf_th,
    image_size,
    slice_height,
    slice_width,
    overlap_height_ratio,
    overlap_width_ratio,
):
    cap, out = load_video(image_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    index = 1
    for _ in tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            break

        boxes = sahi_sliced_predict(
            image_path=frame,
            detection_model_type=detection_model_type,  # yolov8, detectron2, mmdetection, torchvision
            detection_model_path=detection_model_path,
            conf_th=conf_th,
            image_size=image_size,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        if len(boxes) == 0:
            boxes = [0, 0, 0, 0]


        masks = SahiAutoSegmentation().predict(
            source=frame,
            model_type=sam_model_type,
            input_box=boxes,
            multimask_output=False,
            random_color=False,
            show=False,
        )
        a = np.full((1, height, width), False)
        for idx, mask in enumerate(masks):
            if mask[0] != None:
                bmasks = mask.detach().cpu().numpy()
                mask2 = np.logical_or(a, bmasks)
                a[:, :, :] = mask2
        h, w = a.shape[-2:]
        mask_image = a.reshape(h, w, 1)
        cv2.imwrite('./output/image/image%s.jpg' % index, frame)
        cv2.imwrite("output/mask/mask%s.jpg" % index, mask_image * 255)
        mask_image = mask_image * frame
        tmp = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(mask_image)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        cv2.imwrite("output/images/images%s.png" % index, dst)

        lamaindex = lamamain("output/image/image%s.jpg" % index, "output/mask/mask%s.jpg" % index,selected_lamamodel)
        lamaimage = cv2.cvtColor(lamaindex, cv2.COLOR_BGR2RGB)
        cv2.imwrite('output/lama/imagemask%s.jpg' % index, lamaimage)
        index += 1
    list = os.listdir("./output/lama/")
    list.sort(key=lambda x: int(x.replace("imagemask", "").split('.')[0]))
    list1 = os.listdir("./output/images/")
    list1.sort(key=lambda x: int(x.replace("images", "").split('.')[0]))
    frame_width1 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out1 = cv2.VideoWriter("output1.mp4", fourcc, fps, (frame_width1, frame_height1))
    ## 使用切片将图片名称单独切开

    for path in list:
        paths = ("./output/lama/" + path)
        frame = cv2.imread(paths)
        out.write(frame)

    for path in list1:
        paths = ("./output/images/" + path)
        frame = cv2.imread(paths)
        out1.write(frame)
    cap.release()
    return "output.mp4", "output1.mp4"


def change_video(video,bgvideo):
    cap_img = cv2.VideoCapture(video)
    fps = cap_img.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_img.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_img.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_img.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter("output2.mp4",
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (width, height))
    cap_bg = cv2.VideoCapture(bgvideo)
    bg_frame_nums = cap_bg.get(cv2.CAP_PROP_FRAME_COUNT)
    bg_frame_idx = 1
    with tqdm(total=total_frames) as pbar:
        img_frame_idx = 1
        while cap_img.isOpened():
            ret_img, origin_img = cap_img.read()
            if not ret_img:
                break
            ret_bg, bg = cap_bg.read()
            img = cv2.imread("output/mask/mask%s.jpg" % img_frame_idx)
            _, mask_thr = cv2.threshold(img, 240, 1, cv2.THRESH_BINARY)
            if not ret_bg:
                break
            bg_frame_idx += 1
            if bg_frame_idx == bg_frame_nums:
                bg_frame_idx = 1
                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bg = cv2.resize(bg, (width, height))
            if bg.ndim == 2:
                bg = bg[..., np.newaxis]
            out = (mask_thr * origin_img + (1 - mask_thr) * bg).astype(np.uint8)
            cap_out.write(out)
            img_frame_idx += 1
            pbar.update(1)
        cap_img.release()
        cap_out.release()
    return "output2.mp4"

def change_image(video,bgimage):
    cap_img = cv2.VideoCapture(video)
    fps = cap_img.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_img.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_img.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_img.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter("output3.mp4",
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (width, height))
    bg=cv2.imread(bgimage)
    with tqdm(total=total_frames) as pbar:
        img_frame_idx = 1
        while cap_img.isOpened():
            ret_img, origin_img = cap_img.read()
            if not ret_img:
                break
            img = cv2.imread("output/mask/mask%s.jpg" % img_frame_idx)
            _, mask_thr = cv2.threshold(img, 240, 1, cv2.THRESH_BINARY)
            bg = cv2.resize(bg, (width, height))
            if bg.ndim == 2:
                bg = bg[..., np.newaxis]
            out = (mask_thr * origin_img + (1 - mask_thr) * bg).astype(np.uint8)
            cap_out.write(out)
            img_frame_idx += 1
            pbar.update(1)
        cap_img.release()
        cap_out.release()
    return "output3.mp4"

def change_aimage(image,bgimage):
    origin_img=cv2.imread(image)
    bg=cv2.imread(bgimage)
    img = cv2.imread("output/amask/mask.jpg")
    _, mask_thr = cv2.threshold(img, 240, 1, cv2.THRESH_BINARY)
    width = origin_img.shape[1]
    height = origin_img.shape[0]
    bg = cv2.resize(bg, (width, height))
    if bg.ndim == 2:
        bg = bg[..., np.newaxis]
    out = (mask_thr * origin_img + (1 - mask_thr) * bg).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out
def change_avideo(image,bgvideo):
    origin_img = cv2.imread(image)

    cap_bg = cv2.VideoCapture(bgvideo)
    fps = cap_bg.get(cv2.CAP_PROP_FPS)
    bg_frame_nums = cap_bg.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_bg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_bg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter("output4.mp4",
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (width, height))
    mask = cv2.imread("output/amask/mask.jpg")
    origin_img = cv2.resize(origin_img, (width, height))
    mask = cv2.resize(mask, (width, height))
    for _ in tqdm(range(total_frames)):
        ret_bg, bg = cap_bg.read()
        if not ret_bg:
            break
        _, mask_thr = cv2.threshold(mask, 240, 1, cv2.THRESH_BINARY)
        out = (mask_thr * origin_img + (1 - mask_thr) * bg).astype(np.uint8)
        cap_out.write(out)
    cap_bg.release()
    cap_out.release()
    return "output4.mp4"