from ast import arg
import os
import logging
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model import MyVAE
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

SEED = 8
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

logger = TensorBoardLogger(name="logs", save_dir="./")

def main(hparams):
    print("loading model...")
    model = MyVAE(hparams)
    print("model built")
    early_stop = EarlyStopping(
        monitor="val_loss_valid_epoch", patience=5, verbose=True, mode="min"
    )
    checkpoint = ModelCheckpoint(
        dirpath="./ckpt/",
        filename="{}".format(hparams.data_name),
        monitor="val_loss_valid_epoch",
        mode="min",
    )
    trainer = Trainer(
        max_epochs=hparams.max_epoch,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator="gpu",
        gpus=[hparams.gpu],
        check_val_every_n_epoch=1,
        gradient_clip_algorithm='value',
        gradient_clip_val=2,
    )
    print("fit start")
    train_loader = model.mydataloader("train")
    val_loader = model.mydataloader("valid")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=model.mydataloader("test"))
    print("View tensorboard logs by running\ntensorboard --logdir %s" % os.getcwd())
    print("and going to http://localhost:6006 on your browser")


if __name__ == "__main__":
    parser = MyVAE.add_model_specific_args()
    hyperparams = parser.parse_args()
    print(f"RUNNING")
    if hyperparams.only_test == 1:
        model = MyVAE.load_from_checkpoint(checkpoint_path=hyperparams.ckpt_path)
        trainer = Trainer(accelerator="gpu", devices=1)
        trainer.test(model, dataloaders=model.mydataloader("test"))
    else:
        main(hyperparams)