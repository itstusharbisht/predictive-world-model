"""
Train the first executable Predictive World Model prototype.

Expected dataset structure:

data/
    video_001.mp4
    video_002.mp4
    ...

The dataset samples consecutive frame pairs:

    I_t
    I_(t+1)

The model learns to predict the latent representation of the
high-resolution region selected by its own observation query.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from pwm.models import PWMModel


# ============================================================================
# Configuration
# ============================================================================

DATA_DIRECTORY = Path("data")

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

LATENT_DIMENSION = 256

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EPOCHS = 20

FOVEA_SIZE = 128
FOVEA_RADIUS = 0.25

SEARCH_TEMPERATURE = 0.1

NUM_WORKERS = 0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# Video Dataset
# ============================================================================

class ConsecutiveVideoFrames(Dataset):

    def __init__(
        self,
        directory: Path,
        height: int,
        width: int,
    ) -> None:

        self.height = height
        self.width = width

        self.samples: list[
            tuple[Path, int]
        ] = []

        video_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
        }

        videos = [
            path
            for path in directory.rglob("*")
            if path.suffix.lower() in video_extensions
        ]

        if not videos:
            raise RuntimeError(
                f"No videos found in {directory.resolve()}"
            )

        for video in videos:

            capture = cv2.VideoCapture(
                str(video)
            )

            frame_count = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            capture.release()

            for frame_index in range(
                max(0, frame_count - 1)
            ):
                self.samples.append(
                    (
                        video,
                        frame_index,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frame(
        self,
        video: Path,
        index: int,
    ) -> Tensor:

        capture = cv2.VideoCapture(
            str(video)
        )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            index,
        )

        success, frame = capture.read()

        capture.release()

        if not success:
            raise RuntimeError(
                f"Could not read frame {index} from {video}"
            )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = cv2.resize(
            frame,
            (
                self.width,
                self.height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        tensor = torch.from_numpy(
            frame
        )

        tensor = tensor.permute(
            2,
            0,
            1,
        )

        tensor = tensor.float() / 255.0

        return tensor

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:

        video, frame_index = self.samples[index]

        frame_t = self._read_frame(
            video,
            frame_index,
        )

        frame_next = self._read_frame(
            video,
            frame_index + 1,
        )

        return frame_t, frame_next


# ============================================================================
# Loss
# ============================================================================

def prediction_loss(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:

    predicted = F.normalize(
        predicted,
        dim=-1,
    )

    target = F.normalize(
        target.detach(),
        dim=-1,
    )

    return (
        2
        -
        2
        * (
            predicted
            * target
        ).sum(dim=-1)
    ).mean()


# ============================================================================
# Attention Regularization
# ============================================================================

def attention_entropy(
    weights: Tensor,
) -> Tensor:

    weights = weights.clamp_min(
        1e-8
    )

    entropy = -(
        weights
        * weights.log()
    ).sum(dim=-1)

    return entropy.mean()


# ============================================================================
# Training
# ============================================================================

def train() -> None:

    torch.manual_seed(42)
    random.seed(42)

    dataset = ConsecutiveVideoFrames(
        directory=DATA_DIRECTORY,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    model = PWMModel(
        latent_dim=LATENT_DIMENSION,
        crop_size=FOVEA_SIZE,
        fovea_radius=FOVEA_RADIUS,
        search_temperature=SEARCH_TEMPERATURE,
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    for epoch in range(EPOCHS):

        running_loss = 0.0

        for step, (
            frame_t,
            frame_next,
        ) in enumerate(loader):

            frame_t = frame_t.to(
                DEVICE,
                non_blocking=True,
            )

            frame_next = frame_next.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                frame_t,
                frame_next,
            )

            loss_pred = prediction_loss(
                output.predicted_latent,
                output.target_latent,
            )

            # Very small entropy term prevents the initial search
            # distribution from remaining completely diffuse.
            loss_entropy = attention_entropy(
                output.attention_weights
            )

            loss = (
                loss_pred
                +
                0.001
                * loss_entropy
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 0:

                fixation = (
                    output.fixation_xy
                    .detach()
                    .mean(dim=0)
                    .cpu()
                )

                print(
                    f"epoch={epoch + 1:03d} "
                    f"step={step:06d} "
                    f"loss={loss.item():.6f} "
                    f"pred={loss_pred.item():.6f} "
                    f"fixation=("
                    f"{fixation[0]:+.3f}, "
                    f"{fixation[1]:+.3f})"
                )

        epoch_loss = (
            running_loss
            /
            max(1, len(loader))
        )

        print(
            f"Epoch {epoch + 1} "
            f"mean loss: "
            f"{epoch_loss:.6f}"
        )

    Path("checkpoints").mkdir(
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "checkpoints/pwm_v1.pt",
    )


if __name__ == "__main__":
    train()