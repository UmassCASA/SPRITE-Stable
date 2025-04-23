from lightning.pytorch.callbacks import ModelCheckpoint


def get_train_g_loss_checkpoint(dirpath, modelname):
    return ModelCheckpoint(
        monitor="train_g_loss",
        dirpath=dirpath,
        filename=modelname + "-{epoch:03d}-{train_g_loss:.7f}",
        verbose=True,
        save_top_k=1,
        mode="min",
    )


def get_val_g_loss_checkpoint(dirpath, modelname):
    return ModelCheckpoint(
        monitor="val_g_loss",
        dirpath=dirpath,
        filename=modelname + "-{epoch:03d}-{val_g_loss:.7f}",
        verbose=True,
        save_top_k=1,
        mode="min",
    )


def get_checkpoint_callback_nth_epochs(dirpath, modelname, every_n_epochs=250, save_top_k=-1):
    return ModelCheckpoint(
        every_n_epochs=every_n_epochs,
        dirpath=dirpath,
        filename=modelname + "-{epoch:03d}",
        save_top_k=save_top_k,  # Save all checkpoints without restricting to top-k
        verbose=True,
    )


def get_last_epoch_checkpoint(dirpath, modelname):
    return ModelCheckpoint(
        save_top_k=1,
        dirpath=dirpath,
        filename=modelname + "-last-{epoch:03d}",
        verbose=True,
    )
