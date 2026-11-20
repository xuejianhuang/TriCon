import os
import sys
import torch
from argparse import Namespace
from pathlib import Path

from ..config import resolve_path


def load_speech_model(checkpoint_path):
    from fairseq import checkpoint_utils
    import fairseq.utils as fairseq_utils

    checkpoint_path = resolve_path(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    av_hubert_path = resolve_path('av_hubert')

    if not os.path.exists(av_hubert_path):
        raise FileNotFoundError(f"av_hubert directory not found at: {av_hubert_path}")

    sys.path.insert(0, av_hubert_path)
    sys.path.insert(0, os.path.join(av_hubert_path, 'fairseq'))

    try:
        from avhubert import hubert_pretraining
    except ImportError as e:
        print(f"Warning: Could not import av_hubert task: {e}")

    user_dir = os.getcwd()
    fairseq_utils.import_user_module(Namespace(user_dir=user_dir))

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = models[0]

    if hasattr(models[0], 'decoder'):
        model = models[0].encoder.w2v_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, task


def load_dynamic_model(model_path):
    from torchvision.transforms import Compose, CenterCrop

    model_path = resolve_path(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    dynamic_project_root = resolve_path('dynamic')

    if not os.path.exists(dynamic_project_root):
        raise FileNotFoundError(f"dynamic directory not found at: {dynamic_project_root}")

    sys.path.insert(0, dynamic_project_root)

    try:
        from data.transforms import NormalizeVideo, ToTensorVideo
        from models.spatiotemporal_net import get_model as get_dynamic_model
    except ImportError as e:
        raise ImportError(f"Failed to import DynamicForensics modules from {dynamic_project_root}: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_dynamic_model(weights_forgery_path=model_path, device=device)
    model.eval()

    transform = Compose([
        ToTensorVideo(),
        CenterCrop((88, 88)),
        NormalizeVideo((0.421,), (0.165,))
    ])

    return model, transform


def load_lip_model(model_path):
    from torchvision.transforms import Compose, CenterCrop, Normalize

    model_path = resolve_path(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LipForensics model not found: {model_path}")

    dynamic_project_root = resolve_path('dynamic')

    if not os.path.exists(dynamic_project_root):
        raise FileNotFoundError(f"dynamic directory not found at: {dynamic_project_root}")

    sys.path.insert(0, dynamic_project_root)

    try:
        from data.transforms import NormalizeVideo, ToTensorVideo
        from models.spatiotemporal_net import get_model as get_lip_model
    except ImportError as e:
        raise ImportError(f"Failed to import LipForensics modules from {dynamic_project_root}: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_lip_model(weights_forgery_path=model_path, device=device)
    model.eval()

    transform = Compose([
        ToTensorVideo(),
        CenterCrop((88, 88)),
        NormalizeVideo((0.421,), (0.165,))
    ])

    return model, transform
