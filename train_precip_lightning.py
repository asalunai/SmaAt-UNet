import torch
import torchsummary
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models import unet_precip_regression_lightning as unet_regr


def get_batch_size(hparams):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    trainer = pl.Trainer(gpus=hparams.gpus)
    new_batch_size = trainer.scale_batch_size(net, mode='binsearch', init_val=8)
    print("New biggest batch_size: ", new_batch_size)
    return new_batch_size


def train_regression(hparams):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    torchsummary.summary(net, (12, 288, 288), device="cpu")

    default_save_path = "lightning/precip_regression"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd() + "/" + default_save_path + "/" + net.__class__.__name__,
        filename="{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=net.__class__.__name__ + "_rain_threshhold_50_"
    )
    lr_logger = LearningRateLogger()
    tb_logger = TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=hparams.es_patience,  # is effectively half (due to a bug in pytorch-lightning)
    )
    trainer = pl.Trainer(
        fast_dev_run=hparams.fast_dev_run,
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_logger],
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        val_check_interval=hparams.val_check_interval,
        overfit_pct=hparams.overfit_pct,
        logger=tb_logger
    )
    trainer.fit(net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--dataset_folder',
                        default='data/precipitation/RAD_NL25_RAC_5min_train_test_
