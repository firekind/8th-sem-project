import os
from typing import Tuple


def update_attrs(obj, kwargs):
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


class YoloV3Options:
    data: str
    epochs: int
    batch_size: int
    img_size: Tuple[int, int, int]  # [min_train, max-train, test]
    multi_scale: bool = False  # adjust (67% - 150%) img_size every 10 batches
    cfg: str = "config/yolov3.cfg"
    rect: bool = False  # rectangular training
    resume: bool = False  # resume training from last.pt
    nosave: bool = False  # only save final checkpoint
    notest: bool = False  # only test final epoch
    evolve: bool = False  # evolve hyperparameters
    bucket: str = ""  # gsutil bucket
    cache_images: bool = False  # cache images for faster training
    weights: str = "weights/yolov3.weights"
    name: str = ""  # renames results.txt to results_name.txt if supplied
    device: str = ""  # device id (i.e. 0 or 0,1 or cpu)
    adam: bool = False  # use adam optimizer
    single_cls: bool = False  # train as single-class dataset
    freeze_layers: bool = False  # Freeze non-output layers
    gr: float = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    conf_thres: float = 0.001
    iou_thres: float = 0.6  # for nms
    mosiac: bool = False  # apply recap kind of augmentation

    def __init__(self, epochs, batch_size, data, img_size_min = 320, img_size_max = 640, img_size_test = 640, **kwargs):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = (img_size_min, img_size_max, img_size_test)

        update_attrs(self, kwargs)


class YoloHyp:
    giou = 3.54  # giou loss gain
    cls = 37.4  # cls loss gain
    cls_pw = 1.0  # cls BCELoss positive_weight
    obj = 64.3  # obj loss gain (*=img_size/320 if img_size != 320)
    obj_pw = 1.0  # obj BCELoss positive_weight
    iou_t = 0.20  # iou training threshold
    lr0 = 0.01  # initial learning rate (SGD=5E-3 Adam=5E-4)
    lrf = 0.0005  # final learning rate (with cos scheduler)
    momentum = 0.937  # SGD momentum
    weight_decay = 0.0005  # optimizer weight decay
    fl_gamma = 0.0  # focal loss gamma (efficientDet default is gamma=1.5)
    hsv_h = 0.0138  # image HSV-Hue augmentation (fraction)
    hsv_s = 0.678  # image HSV-Saturation augmentation (fraction)
    hsv_v = 0.36  # image HSV-Value augmentation (fraction)
    degrees = 1.98 * 0  # image rotation (+/- deg)
    translate = 0.05 * 0  # image translation (+/- fraction)
    scale = 0.05 * 0  # image scale (+/- gain)
    shear = 0.641 * 0  # image shear (+/- deg)

    def __init__(self, **kwargs):
        update_attrs(self, kwargs)

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class YoloV3Config:
    CONFIG_PATH = "config/yolov3.cfg"
    WEIGHTS_PATH = "weights/yolov3.weights"

    hyp: YoloHyp
    opt: YoloV3Options

    def __init__(self, **kwargs):
        update_attrs(self, kwargs)
        self.hyp = YoloHyp(**kwargs)
        self.opt = YoloV3Options(**kwargs)


class Config:
    EPOCHS = 100
    BATCH_SIZE = 16
    DATA_DIR = "data/mini"
    IMG_SIZE = 640
    MIN_IMG_SIZE = 480  # used in yolo and planercnn
    YOLO_HEAD_LR = 1e-4
    LOG_RES_EVERY_N_BATCHES = 2

    yolo_config: YoloV3Config

    def __init__(self, **kwargs):
        update_attrs(self, kwargs)

        kwargs["epochs"] = self.EPOCHS
        kwargs["batch_size"] = self.BATCH_SIZE
        kwargs["image_size_min"] = self.MIN_IMG_SIZE
        kwargs["img_size_max"] = self.IMG_SIZE
        kwargs["img_size_test"] = self.IMG_SIZE
        kwargs["IMAGE_MAX_DIM"] = self.IMG_SIZE
        kwargs["IMAGE_MIN_DIM"] = self.MIN_IMG_SIZE

        self.yolo_config = YoloV3Config(**{
            "data": os.path.join(self.DATA_DIR, "custom.data"),
            **kwargs
        })
