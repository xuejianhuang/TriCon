import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from python_speech_features import logfbank
import soundfile as sf
import os.path as osp
import tempfile


def stacker(feats, stack_order):
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
    return feats


def extract_audio_features(audio_path, sample_rate=16000):
    sample_rate_read, wav_data = wavfile.read(audio_path)
    assert sample_rate_read == sample_rate and len(wav_data.shape) == 1

    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
    audio_feats = stacker(audio_feats, 4)

    return audio_feats


def normalize_audio_features(audio_feats):
    audio_feats_tensor = torch.FloatTensor(audio_feats).cuda()
    with torch.no_grad():
        audio_feats_tensor = F.layer_norm(audio_feats_tensor, audio_feats_tensor.shape[1:])
    audio_feats_tensor = audio_feats_tensor.transpose(0, 1).unsqueeze(dim=0)

    return audio_feats_tensor


def truncate_audio(audio_path, max_length=50, sample_rate=16000):
    wav, sr = sf.read(audio_path)
    max_samples = sr * max_length

    if len(wav) > max_samples:
        tmp_dir = tempfile.mkdtemp()
        temp_wav_path = osp.join(tmp_dir, 'audio.wav')
        sf.write(temp_wav_path, wav[:max_samples], sr)
        return temp_wav_path

    return audio_path


def cosine_similarity(feat1, feat2):
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)

    if len(feat1) != len(feat2):
        sample = np.linspace(0, len(feat1)-1, len(feat2), dtype=int)
        feat1 = feat1[sample.tolist()]

    similarity = F.cosine_similarity(feat1, feat2)
    return similarity.cpu().numpy()
