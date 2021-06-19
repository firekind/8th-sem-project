import torch
from src.yolov3.utils.datasets import LoadImagesAndLabels as YoloDataset
from src.yolov3.utils.parse_config import parse_data_cfg as parse_yolo_data_cfg

def dataset(config, train=True):
    data_dict = parse_yolo_data_cfg(config.yolo_config.opt.data)
    return YoloDataset(
        data_dict["train"] if train else data_dict["valid"],
        config.IMG_SIZE if train else config.yolo_config.opt.img_size[-1],
        config.BATCH_SIZE,
        augment=False,
        hyp=config.yolo_config.hyp,
        rect=config.yolo_config.opt.rect if train else True,
        cache_images=config.yolo_config.opt.cache_images,
        single_cls=config.yolo_config.opt.single_cls,
        mosiac=config.yolo_config.opt.mosiac,
        label_files_path=data_dict["labels"],
    )


def collate_fn(batch):
    img, label, path, shapes, pad = zip(*batch)
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()

    return (
        torch.stack(img) / 255,
        (
            torch.stack(img, 0),
            torch.cat(label, 0),
            path,
            shapes,
            pad,
        ),
    )
