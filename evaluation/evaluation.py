import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import librosa
from pesq import pesq
import matplotlib.pyplot as plt
import scipy

from scipy.io.wavfile import write
from IPython.display import Audio

import torch

from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels
from typing import List

# Dataset stuff
from everyvoice.text.lookups import LookupTables
from everyvoice.dataloader import BaseDataModule
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.dataset import (
    FastSpeech2DataModule,
    FastSpeechDataset,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
    FastSpeech2Config,
)
from torch.utils.data import DataLoader

# Preprocessing stuff
from everyvoice.preprocessor import Preprocessor

# Feature prediction stuff
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import (
    FastSpeech2,
)

# Vocoder stuff
from everyvoice.model.vocoder.original_hifigan_helper import (
    get_vocoder,
    vocoder_infer,
)

import sys

sys.path.append("/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec")
sys.path.append("/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/spec2spec")

from spec2spec.dataset import DenoisingDataset
from spec2spec.train import SpecDenoiser
from spec2spec.models import DnCNN, PostNet, DnCNNConfig, PostNetConfig
from spec2spec.utils import plot_pairs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Try these denoisers
# Very bad
# EveryVoice_logs/Segmented/PostNet/PostNet-lr-1e-5-MSE
# EveryVoice_logs/Segmented/PostNet/PostNet-lr-1e-5-L1

# Better
# 60-minutes-ablation/Complete/PostNet/PostNet-lr-1e-5
# 60-minutes-ablation/Complete/PostNet/PostNet-lr-1e-6
# 60-minutes-ablation/Segmented/PostNet/PostNet-lr-1e-6

MODEL_PATH = "/space/partner/nrc/work/dt/eng000/Experiments/test-en/60-minutes/logs/FeaturePredictionExperiment/base/checkpoints-postnet/last.ckpt"
VOCODER_PATH = "/home/eng000/sgile/models/hifigan/generator_universal.pth.tar"
DENOISER_PATH = "/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/spec2spec/EveryVoice_logs/Complete/DnCNN/DnCNN-Gaussian/checkpoints/last.ckpt"

model: FastSpeech2 = FastSpeech2.load_from_checkpoint(MODEL_PATH).to(device)
preprocessor: Preprocessor = Preprocessor(model.config)
model.eval()

vocoder = get_vocoder(VOCODER_PATH, device=device)

denoiser = SpecDenoiser()
denoiser = denoiser.load_from_checkpoint(DENOISER_PATH).to(device)
denoiser.eval()

TEST_PATH = "/space/partner/nrc/work/dt/eng000/Experiments/data/spec-to-spec/EveryVoice/60-minutes-english-ablation/test.txt"
test = pd.read_csv(TEST_PATH, sep="|")

for (basename, text) in tqdm(test.values, desc='Inference'):
    text_tensor = preprocessor.extract_text_inputs(text)
    src_lens = torch.LongTensor([text_tensor.size(0)])
    max_src_len = max(src_lens)
    batch = {
        "text": text_tensor,
        "src_lens": src_lens,
        "max_src_len": max_src_len,
        "speaker_id": torch.LongTensor([0]),  # These are hardcoded but work in training
        "language_id": torch.LongTensor(
            [0]
        ),  # These are hardcoded but work in training
    }
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["max_mel_len"] = 1_000_000
    batch["mel_lens"] = None

    # Run model
    with torch.no_grad():
        output = model.forward(batch, inference=True)

    spec = output["output"][0].transpose(0, 1)
    spec_postnet = output["postnet_output"][0].transpose(0, 1)

    # Original output
    wav = vocoder_infer(
        torch.tensor(spec).transpose(1, 0).unsqueeze(0), vocoder, 32768.0
    )[0]
    write(
        f"/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/output/{basename}.wav",
        22050,
        wav,
    )

    # PostNet output
    wav = vocoder_infer(
        torch.tensor(spec_postnet).transpose(1, 0).unsqueeze(0), vocoder, 32768.0
    )[0]
    write(
        f"/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/postnet-output/{basename}.wav",
        22050,
        wav,
    )

    # Due to the batchnorm I have some troubles running inference on batches of size 1
    batch = torch.stack([spec.unsqueeze(0), spec.unsqueeze(0)])
    denoised = denoiser(batch)[0].squeeze()

    # Denoised output
    wav = vocoder_infer(
        torch.tensor(denoised).transpose(1, 0).unsqueeze(0), vocoder, 32768.0
    )[0]
    write(
        f"/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/denoised-output/{basename}.wav",
        22050,
        wav,
    )

    # Due to the batchnorm I have some troubles running inference on batches of size 1
    batch = torch.stack([spec_postnet.unsqueeze(0), spec_postnet.unsqueeze(0)])
    denoised = denoiser(batch)[0].squeeze()

    # Denoised postnet output
    wav = vocoder_infer(
        torch.tensor(denoised).transpose(1, 0).unsqueeze(0), vocoder, 32768.0
    )[0]
    write(
        f"/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/denoised-postnet-output/{basename}.wav",
        22050,
        wav,
    )


def get_mcd_distances(
    ground_truths: List[Path],
    predictions: List[Path],
    penalty: bool = False,
    n_mels: int = 80,
):
    assert len(ground_truths) == len(predictions)

    # metrics = [
    #     get_metrics_wavs(wav_file_1=ref, wav_file_2=deg)
    #     for (ref, deg) in zip(ground_truths, predictions)
    # ]
    metrics = []
    for (ref, deg) in tqdm(
        zip(ground_truths, predictions), total=len(ground_truths), desc="MCD"
    ):
        try:
            metric = get_metrics_wavs(wav_file_1=ref, wav_file_2=deg)
            metrics.append(metric)
        except Exception as e:
            print(f"{e} in file {ref}")

    distances = np.array([metric[0] for metric in metrics])
    penalties = np.array([metric[1] for metric in metrics])
    final_frames_number = np.array([metric[2] for metric in metrics])

    if penalty:
        distances = distances + penalties

    return round(np.mean(distances), 4)


def get_f0_correlation(ground_truths: List[Path], predictions: List[Path]):
    assert len(ground_truths) == len(predictions)

    # metrics = [
    #     get_metrics_wavs(wav_file_1=ref, wav_file_2=deg)
    #     for (ref, deg) in zip(ground_truths, predictions)
    # ]
    correlations = []
    for (ref, deg) in tqdm(
        zip(ground_truths, predictions), total=len(ground_truths), desc="F0"
    ):
        try:
            ref_wav, sr = librosa.load(ref)
            deg_wav, sr = librosa.load(deg)

            f0_ref = librosa.yin(
                ref_wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
            )
            f0_deg = librosa.yin(
                deg_wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
            )

            # new_shape = min(len(f0_ref), len(f0_deg))
            # f0_ref, f0_deg = f0_ref[0:new_shape], f0_deg[0:new_shape]

            # Calculate the amount of padding for each array
            max_length = max(len(f0_ref), len(f0_deg))
            pad_length1 = max_length - len(f0_ref)
            pad_length2 = max_length - len(f0_deg)

            # Pad the arrays with zeros
            f0_ref = np.pad(f0_ref, (0, pad_length1), mode="constant")
            f0_deg = np.pad(f0_deg, (0, pad_length2), mode="constant")

            corr = scipy.stats.pearsonr(f0_ref, f0_deg).statistic
            correlations.append(corr)

        except Exception as e:
            print(f"{e} in file {ref}")

    return round(np.mean(correlations), 4)


def get_pesq_score(
    ground_truths: List[Path],
    predictions: List[Path],
    sr: int = 16000,
    mode: str = "wb",
):
    assert len(ground_truths) == len(predictions)

    # metrics = [
    #     get_metrics_wavs(wav_file_1=ref, wav_file_2=deg)
    #     for (ref, deg) in zip(ground_truths, predictions)
    # ]
    scores = []
    for (ref, deg) in tqdm(
        zip(ground_truths, predictions), total=len(ground_truths), desc="PESQ"
    ):
        try:
            ref, _ = librosa.load(ref, sr=sr)
            deg, _ = librosa.load(deg, sr=sr)

            score = pesq(sr, ref, deg, mode=mode)
            scores.append(score)
        except Exception as e:
            print(f"{e} in file {ref}")

    return round(np.mean(scores), 4)


ground_truth_path = (
    "/space/partner/nrc/work/dt/eng000/Experiments/data/LJSpeech-en/wavs"
)
prediction_paths = [
    "/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/output",
    "/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/postnet-output",
    "/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/denoised-output",
    "/space/partner/nrc/work/dt/eng000/Repositories/spec-to-spec/evaluation/denoised-postnet-output",
]

results = pd.DataFrame(
    columns=[
        "Metric",
        "FastSpeech2",
        "FastSpeech2 + postnet",
        "FastSpeech2 + denoiser",
        "FastSpeech2 + postnet + denoiser",
    ]
)

metrics = {"mcd": get_mcd_distances, "f0": get_f0_correlation, "pesq": get_pesq_score}

for metric in metrics:
    new_row = [metric]
    for prediction_path in prediction_paths:
        ground_truths = [
            Path(f"{ground_truth_path}/{basename}.wav") for basename in test["basename"]
        ]
        predictions = [
            Path(f"{prediction_path}/{basename}.wav") for basename in test["basename"]
        ]

        score = metrics.get(metric)(ground_truths, predictions)
        new_row.append(score)

    results.loc[len(results)] = new_row

print("\n")
print(results)
