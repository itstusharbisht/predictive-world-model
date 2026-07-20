"""
Evaluation for the first Predictive World Model (PWM) prototype.

Produces:
    outputs/attention_video.mp4
    outputs/fixation_coordinates.csv
    outputs/foveal_crops/

Evaluates:
    - fixation trajectory
    - fixation variance
    - attention entropy
    - frame-to-frame saccade distance
    - latent prediction loss

The rendered video shows:
    - predicted fovea location
    - coarse attention heatmap
    - recent fixation trajectory
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pwm.models import PWMModel


# ============================================================================
# Configuration
# ============================================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

LATENT_DIMENSION = 256

FOVEA_SIZE = 128
FOVEA_RADIUS = 0.25

SEARCH_TEMPERATURE = 0.1

CHECKPOINT_PATH = Path("checkpoints/pwm_v1.pt")

OUTPUT_DIRECTORY = Path("outputs")

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# Frame Conversion
# ============================================================================

def frame_to_tensor(
    frame_bgr: np.ndarray,
) -> Tensor:
    """
    Convert an OpenCV BGR frame into the normalized tensor format expected
    by PWM.

    Returns:
        [1, 3, H, W]
    """

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    frame_rgb = cv2.resize(
        frame_rgb,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    tensor = torch.from_numpy(
        frame_rgb
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    tensor = tensor.float() / 255.0

    return tensor.unsqueeze(0)


# ============================================================================
# Coordinate Conversion
# ============================================================================

def normalized_to_pixel(
    fixation_xy: Tensor,
    width: int,
    height: int,
) -> tuple[int, int]:
    """
    Convert normalized coordinates [-1, 1] into image pixel coordinates.
    """

    x_normalized = float(
        fixation_xy[0].item()
    )

    y_normalized = float(
        fixation_xy[1].item()
    )

    x = int(
        ((x_normalized + 1.0) / 2.0)
        * (width - 1)
    )

    y = int(
        ((y_normalized + 1.0) / 2.0)
        * (height - 1)
    )

    x = max(
        0,
        min(width - 1, x),
    )

    y = max(
        0,
        min(height - 1, y),
    )

    return x, y


# ============================================================================
# Metrics
# ============================================================================

def prediction_loss(
    predicted: Tensor,
    target: Tensor,
) -> float:

    predicted = F.normalize(
        predicted,
        dim=-1,
    )

    target = F.normalize(
        target,
        dim=-1,
    )

    loss = (
        2
        -
        2
        * (
            predicted
            * target
        ).sum(dim=-1)
    ).mean()

    return float(loss.item())


def attention_entropy(
    weights: Tensor,
) -> float:

    weights = weights.clamp_min(
        1e-8
    )

    entropy = -(
        weights
        * weights.log()
    ).sum(dim=-1)

    return float(
        entropy.mean().item()
    )


def normalized_entropy(
    weights: Tensor,
) -> float:
    """
    Returns entropy normalized approximately to [0, 1].

    1:
        nearly uniform attention

    0:
        highly concentrated attention
    """

    number_of_locations = weights.shape[-1]

    entropy = attention_entropy(
        weights
    )

    maximum_entropy = np.log(
        number_of_locations
    )

    if maximum_entropy <= 0:
        return 0.0

    return float(
        entropy / maximum_entropy
    )


# ============================================================================
# Attention Heatmap
# ============================================================================

def build_attention_heatmap(
    weights: Tensor,
    coarse_height: int,
    coarse_width: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:

    attention = (
        weights[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    attention = attention.reshape(
        coarse_height,
        coarse_width,
    )

    minimum = attention.min()
    maximum = attention.max()

    attention = (
        attention - minimum
    ) / (
        maximum
        - minimum
        + 1e-8
    )

    attention = (
        attention * 255.0
    ).astype(
        np.uint8
    )

    attention = cv2.resize(
        attention,
        (
            output_width,
            output_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    return cv2.applyColorMap(
        attention,
        cv2.COLORMAP_JET,
    )


# ============================================================================
# Foveal Crop Saving
# ============================================================================

def save_foveal_crop(
    crop: Tensor,
    output_path: Path,
) -> None:

    image = (
        crop[0]
        .detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
    )

    image = (
        image
        .permute(1, 2, 0)
        .numpy()
    )

    image = (
        image * 255.0
    ).astype(
        np.uint8
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )


# ============================================================================
# Model Loading
# ============================================================================

def load_model() -> PWMModel:

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH.resolve()}"
        )

    model = PWMModel(
        latent_dim=LATENT_DIMENSION,
        crop_size=FOVEA_SIZE,
        fovea_radius=FOVEA_RADIUS,
        search_temperature=SEARCH_TEMPERATURE,
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    video_path: Path,
) -> None:

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: "
            f"{video_path.resolve()}"
        )

    OUTPUT_DIRECTORY.mkdir(
        exist_ok=True
    )

    crop_directory = (
        OUTPUT_DIRECTORY
        / "foveal_crops"
    )

    crop_directory.mkdir(
        exist_ok=True
    )

    csv_path = (
        OUTPUT_DIRECTORY
        / "fixation_coordinates.csv"
    )

    video_output_path = (
        OUTPUT_DIRECTORY
        / "attention_video.mp4"
    )

    model = load_model()

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    success, previous_frame = (
        capture.read()
    )

    if not success:

        capture.release()

        raise RuntimeError(
            "Video contains no readable frames."
        )

    previous_frame = cv2.resize(
        previous_frame,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ),
    )

    writer = cv2.VideoWriter(
        str(video_output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ),
    )

    fixation_history: deque[
        tuple[int, int]
    ] = deque(
        maxlen=30
    )

    all_fixations: list[
        tuple[float, float]
    ] = []

    all_prediction_losses: list[
        float
    ] = []

    all_entropies: list[
        float
    ] = []

    all_saccade_distances: list[
        float
    ] = []

    previous_fixation: tuple[
        float,
        float
    ] | None = None

    frame_index = 0

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        csv_writer = csv.writer(
            csv_file
        )

        csv_writer.writerow(
            [
                "frame",
                "x_normalized",
                "y_normalized",
                "x_pixel",
                "y_pixel",
                "prediction_loss",
                "attention_entropy",
                "normalized_entropy",
                "saccade_distance",
            ]
        )

        while True:

            success, current_frame = (
                capture.read()
            )

            if not success:
                break

            current_frame = cv2.resize(
                current_frame,
                (
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                ),
            )

            frame_t = frame_to_tensor(
                previous_frame
            ).to(
                DEVICE
            )

            frame_next = frame_to_tensor(
                current_frame
            ).to(
                DEVICE
            )

            with torch.no_grad():

                output = model(
                    frame_t,
                    frame_next,
                )

                coarse_next = (
                    model.coarse_encoder(
                        frame_next
                    )
                )

            fixation = (
                output.fixation_xy[0]
                .detach()
                .cpu()
            )

            x_normalized = float(
                fixation[0].item()
            )

            y_normalized = float(
                fixation[1].item()
            )

            x_pixel, y_pixel = (
                normalized_to_pixel(
                    fixation,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                )
            )

            loss = prediction_loss(
                output.predicted_latent,
                output.target_latent,
            )

            entropy = attention_entropy(
                output.attention_weights
            )

            norm_entropy = (
                normalized_entropy(
                    output.attention_weights
                )
            )

            if previous_fixation is None:

                saccade_distance = 0.0

            else:

                dx = (
                    x_normalized
                    - previous_fixation[0]
                )

                dy = (
                    y_normalized
                    - previous_fixation[1]
                )

                saccade_distance = float(
                    np.sqrt(
                        dx * dx
                        +
                        dy * dy
                    )
                )

            previous_fixation = (
                x_normalized,
                y_normalized,
            )

            all_fixations.append(
                (
                    x_normalized,
                    y_normalized,
                )
            )

            all_prediction_losses.append(
                loss
            )

            all_entropies.append(
                norm_entropy
            )

            all_saccade_distances.append(
                saccade_distance
            )

            fixation_history.append(
                (
                    x_pixel,
                    y_pixel,
                )
            )

            csv_writer.writerow(
                [
                    frame_index,
                    x_normalized,
                    y_normalized,
                    x_pixel,
                    y_pixel,
                    loss,
                    entropy,
                    norm_entropy,
                    saccade_distance,
                ]
            )

            # -----------------------------------------------------------
            # Attention heatmap
            # -----------------------------------------------------------

            coarse_height = (
                coarse_next.shape[-2]
            )

            coarse_width = (
                coarse_next.shape[-1]
            )

            heatmap = (
                build_attention_heatmap(
                    output.attention_weights,
                    coarse_height,
                    coarse_width,
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                )
            )

            visualization = (
                cv2.addWeighted(
                    current_frame,
                    0.72,
                    heatmap,
                    0.28,
                    0.0,
                )
            )

            # -----------------------------------------------------------
            # Fixation trajectory
            # -----------------------------------------------------------

            history = list(
                fixation_history
            )

            for index in range(
                1,
                len(history),
            ):

                cv2.line(
                    visualization,
                    history[index - 1],
                    history[index],
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                )

            # -----------------------------------------------------------
            # Foveal radius
            # -----------------------------------------------------------

            radius_pixels = int(
                FOVEA_RADIUS
                * IMAGE_WIDTH
            )

            cv2.circle(
                visualization,
                (
                    x_pixel,
                    y_pixel,
                ),
                radius_pixels,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

            cv2.circle(
                visualization,
                (
                    x_pixel,
                    y_pixel,
                ),
                6,
                (
                    255,
                    255,
                    255,
                ),
                -1,
            )

            # -----------------------------------------------------------
            # Metrics overlay
            # -----------------------------------------------------------

            cv2.putText(
                visualization,
                (
                    f"Loss: "
                    f"{loss:.4f}"
                ),
                (
                    15,
                    30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

            cv2.putText(
                visualization,
                (
                    f"Entropy: "
                    f"{norm_entropy:.3f}"
                ),
                (
                    15,
                    60,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

            cv2.putText(
                visualization,
                (
                    f"Saccade: "
                    f"{saccade_distance:.3f}"
                ),
                (
                    15,
                    90,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

            writer.write(
                visualization
            )

            # Save periodic crops instead of every frame.
            if frame_index % 30 == 0:

                save_foveal_crop(
                    output.foveal_crop,
                    crop_directory
                    / (
                        f"frame_"
                        f"{frame_index:06d}.jpg"
                    ),
                )

            previous_frame = (
                current_frame
            )

            frame_index += 1

    capture.release()
    writer.release()

    # ====================================================================
    # Final Metrics
    # ====================================================================

    if not all_fixations:

        raise RuntimeError(
            "No frame pairs were evaluated."
        )

    fixations = np.asarray(
        all_fixations,
        dtype=np.float32,
    )

    fixation_variance_x = float(
        np.var(
            fixations[:, 0]
        )
    )

    fixation_variance_y = float(
        np.var(
            fixations[:, 1]
        )
    )

    total_fixation_variance = (
        fixation_variance_x
        +
        fixation_variance_y
    )

    mean_loss = float(
        np.mean(
            all_prediction_losses
        )
    )

    mean_entropy = float(
        np.mean(
            all_entropies
        )
    )

    mean_saccade = float(
        np.mean(
            all_saccade_distances
        )
    )

    print()
    print(
        "PWM Evaluation"
    )
    print(
        "=============="
    )

    print(
        f"Frames evaluated: "
        f"{frame_index}"
    )

    print(
        f"Mean prediction loss: "
        f"{mean_loss:.6f}"
    )

    print(
        f"Fixation variance X: "
        f"{fixation_variance_x:.6f}"
    )

    print(
        f"Fixation variance Y: "
        f"{fixation_variance_y:.6f}"
    )

    print(
        f"Total fixation variance: "
        f"{total_fixation_variance:.6f}"
    )

    print(
        f"Mean normalized attention entropy: "
        f"{mean_entropy:.6f}"
    )

    print(
        f"Mean saccade distance: "
        f"{mean_saccade:.6f}"
    )

    print()
    print(
        f"Video: "
        f"{video_output_path.resolve()}"
    )

    print(
        f"Coordinates: "
        f"{csv_path.resolve()}"
    )

    print(
        f"Foveal crops: "
        f"{crop_directory.resolve()}"
    )

    # ====================================================================
    # Basic Collapse Warnings
    # ====================================================================

    print()
    print(
        "Diagnostics"
    )
    print(
        "==========="
    )

    if total_fixation_variance < 1e-4:

        print(
            "WARNING: Fixation appears nearly static."
        )

    else:

        print(
            "Fixation changes across the video."
        )

    if mean_entropy > 0.95:

        print(
            "WARNING: Attention is nearly uniform."
        )

    elif mean_entropy < 0.05:

        print(
            "WARNING: Attention is extremely concentrated."
        )

    else:

        print(
            "Attention distribution is non-trivial."
        )


# ============================================================================
# Command Line
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PWM fixation behaviour "
            "on a video."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to evaluation video.",
    )

    args = parser.parse_args()

    evaluate(
        args.video
    )


if __name__ == "__main__":
    main()