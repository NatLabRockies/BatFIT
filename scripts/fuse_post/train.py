import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from classifier import JointBoundaryClassifier
from fuse_utils import prepare_datasets
from torch.utils.data import DataLoader

from batfit import BATFIT_EXP, logger
from batfit.utils.torch_utils import (
    load_model,
    log_training,
    prepare_log,
    save_model,
)


def train_classifier(
    p1_array: np.ndarray,
    p2_array: np.ndarray,
    yaml_path: str,
    hidden_dim: int = 64,
    num_layers: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
) -> JointBoundaryClassifier:

    logger.info("Preparing datasets...")
    dataset1, dataset2, input_dim = prepare_datasets(
        p1_array, p2_array, yaml_path
    )

    # Shuffle=True is critical here to mix the valid/invalid samples
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

    model = JointBoundaryClassifier(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    prepare_log("train_log")
    logger.info("Starting training...")
    model.train()

    total_steps = epochs * len(loader1)
    current_step = 0
    for epoch in range(epochs):
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0

        # Zip allows us to iterate through both dataloaders simultaneously
        for (x1, y1), (x2, y2) in zip(loader1, loader2):
            current_step += 1
            x1, y1 = x1.to(model.device), y1.to(model.device)
            x2, y2 = x2.to(model.device), y2.to(model.device)

            optimizer.zero_grad()

            logits1, logits2 = model(x1, x2)

            loss1 = criterion(logits1, y1)
            loss2 = criterion(logits2, y2)

            # Combine losses to update both networks simultaneously
            total_loss = loss1 + loss2
            total_loss.backward()
            optimizer.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

            # Log loss
            logged = False
            save_freq = 1000
            log_freq = 1000
            if current_step % save_freq == 0:
                logged = True
                log_training(
                    current_step,
                    total_loss,
                    "train_log",
                    filename="train_loss.csv",
                )
                save_model(
                    step=current_step,
                    model=model,
                    optimizer=optimizer,
                    device_type=None,
                    log_folder="train_log",
                )
            elif current_step % log_freq == 0 and not logged:
                log_training(
                    current_step, loss, "train_log", filename="train_loss.csv"
                )

        avg_loss1 = epoch_loss1 / len(loader1)
        avg_loss2 = epoch_loss2 / len(loader2)

        if (epoch + 1) % 1 == 0 or epoch == 0 or current_step % 1000 == 0:
            logger.info(
                f"Step {current_step}/{total_steps} Ep {epoch+1}/{epochs} | Net1 Loss: {avg_loss1:.4f} | Net2 Loss: {avg_loss2:.4f}"
            )

        logged = False
    save_model(
        step=current_step,
        model=model,
        optimizer=optimizer,
        bypass="final",
        device_type=None,
        log_folder="train_log",
    )
    logger.info("Training complete.")
    return model


if __name__ == "__main__":
    # --- Example Usage ---
    # Dummy data generation representing your stable regions
    p1_samples = np.load("data/assembled_data_diffcap.npz")["Y_data"]
    p2_samples = np.load("data/assembled_data_hppc.npz")["Y_data"]
    config = os.path.join(BATFIT_EXP, "p2d_diffcap.yaml")
    trained_model = train_classifier(
        p1_array=p1_samples,
        p2_array=p2_samples,
        yaml_path=config,
        hidden_dim=256,
        num_layers=2,
        batch_size=512,
        lr=0.001,
        epochs=50,
    )
