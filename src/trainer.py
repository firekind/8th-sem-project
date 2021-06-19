from typing import Dict, List, Any

import numpy as np
import torch.optim as optim

from src.yolov3.eva5_helper import YoloTrainer
from src.yolov3.models import Darknet, load_darknet_weights
from src.yolov3.utils.torch_utils import ModelEMA
from src.yolov3.utils.utils import (
    labels_to_class_weights,
    non_max_suppression,
    output_to_target,
)
from src.utils import plot_yolo_bbox

import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, config, num_batches, num_classes, yolo_labels):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.config = config
        yolo_config = self.config.yolo_config

        self.yolo = Darknet(yolo_config.CONFIG_PATH)
        if yolo_config.WEIGHTS_PATH is not None:
            load_darknet_weights(self.yolo, yolo_config.WEIGHTS_PATH)

        self.yolo.nc = (
            num_classes  # attach number of classes to model
        )
        self.yolo.hyp = (
            yolo_config.hyp
        )  # attach hyperparameters to model
        self.yolo.gr = (
            1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        )
        self.yolo.class_weights = labels_to_class_weights(
            yolo_labels, num_classes
        )  # attach class weights

        self.yolo_trainer = YoloTrainer(
            self.yolo,
            yolo_config.hyp,
            yolo_config.opt,
            num_batches,
            num_classes,
        )

        self.yolo_ema = ModelEMA(self.yolo)
        self.is_ema_on_device = False

    def forward(self, x, yolo_ema=None):
        if self.training:
            yolo_out = self.yolo(x)
        else:
            yolo_out = yolo_ema(x)

        return yolo_out

    def training_step(self, batch, batch_idx):
        imgs, yolo_data = batch

        yolo_out = self(imgs)
        loss, _ = self.yolo_trainer.post_train_step(yolo_out, yolo_data, batch_idx, self.current_epoch)

        # backward and stepping optimizer
        optimizer = self.optimizers()
        self.manual_backward(loss, optimizer)
        if (self.yolo_trainer.calc_ni(batch_idx, self.current_epoch) % self.yolo_trainer.accumulate == 0):
            self.manual_optimizer_step(optimizer)
            self.yolo_ema.update(self.yolo)
        else:
            self.manual_optimizer_step(optimizer)

        self.log("yolo loss", loss.item(), prog_bar=True)
        metrics = {"yolo_loss": loss.item()}

        return metrics

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_yolo_loss = np.mean([d["yolo_loss"] for d in outputs])
        self.log("avg yolo loss", avg_yolo_loss, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.yolo_trainer.validation_epoch_start()
        self.yolo_ema.update_attr(self.yolo)

    def validation_step(self, batch, batch_idx):
        imgs, yolo_data = batch
        metrics = {}

        if not self.is_ema_on_device:
            self.yolo_ema.ema = self.yolo_ema.ema.to(self.device)
            self.is_ema_on_device = True

        ema = self.yolo_ema.ema
    
        yolo_out = self(imgs, yolo_ema=ema)

        yolo_losses = self.yolo_trainer.validation_step(
            self.config.yolo_config.opt,
            yolo_out,
            yolo_data,
            batch_idx,
            self.current_epoch,
        )

        metrics.update({"yolo_val_losses": yolo_losses.sum().item()})

        # logging intermediate results
        if batch_idx % self.config.LOG_RES_EVERY_N_BATCHES:
            # logging yolo outputs
            inf_out, _ = yolo_out
            output = non_max_suppression(
                inf_out,
                conf_thres=self.config.yolo_config.opt.conf_thres,
                iou_thres=self.config.yolo_config.opt.iou_thres,
            )
            res_img = plot_yolo_bbox(
                yolo_data[0],
                output_to_target(output, imgs.shape[-1], imgs.shape[-2]),
                names=["hardhat", "vest", "mask", "boots"],
            )

            self.logger.experiment.add_image(
                "yolo outputs", res_img, self.current_epoch, dataformats="HWC"
            )

        return metrics

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # `mAPs` is avg. precision for each class, `mAP` is total mAP
        (mp, mr, mAP, mf1), mAPs = self.yolo_trainer.validation_epoch_end()
        avg_yolo_val_loss = np.mean([d["yolo_val_losses"] for d in outputs])

        # log stuff
        self.log("avg yolo val loss", avg_yolo_val_loss, prog_bar=True)
        self.log("yolo mAP", mAP, prog_bar=True)
        self.log("yolo mean precision", mp)
        self.log("yolo mean recall", mr)
        self.log("yolo mean f1", mf1)

    def configure_optimizers(self):
        param_groups = []

        # yolo param groups
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.yolo.named_parameters()).items():
            if ".bias" in k:
                pg2 += [v]  # biases
            elif "Conv2d.weight" in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        hyp = self.config.yolo_config.hyp
        param_groups.append(
            {
                "params": pg0,
                "lr": hyp["lr0"],
                "momentum": hyp["momentum"],
                "nesterov": True,
            }
        )
        param_groups.append(
            {
                "params": pg1,
                "lr": hyp["lr0"],
                "momentum": hyp["momentum"],
                "nesterov": True,
                "weight_decay": hyp["weight_decay"],
            }
        )
        param_groups.append(
            {
                "params": pg2,
                "lr": hyp["lr0"],
                "momentum": hyp["momentum"],
                "nesterov": True,
            }
        )

        # creating optimizer
        optimizer = optim.SGD(param_groups)

        # setting the optimizer for the yolo trainer
        self.yolo_trainer.set_optimizer(optimizer)

        return optimizer

    def get_progress_bar_dict(self):
        prog_dict = super().get_progress_bar_dict()
        prog_dict.pop("loss", None)
        return prog_dict

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint.update({"yolo_ema": self.yolo_ema.ema.state_dict()})

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        self.yolo_ema.ema.load_state_dict(checkpoint["yolo_ema"])
