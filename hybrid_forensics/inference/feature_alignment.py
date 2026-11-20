import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def align_temporal_features(
    speech_feat: Dict[str, torch.Tensor],
    lip_feat: Dict[str, torch.Tensor],
    target_len: int = 25,
    use_concat: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_concat:
        if 'concat' not in speech_feat:
            sem_raw = torch.cat([speech_feat['visual'], speech_feat['audio']], dim=-1)
        else:
            sem_raw = speech_feat['concat']
    else:
        sem_raw = speech_feat['visual']

    dyn_raw = lip_feat['temporal']

    sem_T = sem_raw.transpose(0, 1).unsqueeze(0)
    dyn_T = dyn_raw.transpose(0, 1).unsqueeze(0)

    sem_pooled = F.adaptive_avg_pool1d(sem_T, output_size=target_len)
    dyn_pooled = F.adaptive_avg_pool1d(dyn_T, output_size=target_len)

    sem_aligned = sem_pooled.squeeze(0).transpose(0, 1)
    dyn_aligned = dyn_pooled.squeeze(0).transpose(0, 1)

    return sem_aligned, dyn_aligned


def create_padding_mask(
    speech_feat: Dict[str, torch.Tensor],
    lip_feat: Dict[str, torch.Tensor],
    target_len: int = 25,
    frames_per_clip: int = 25
) -> torch.Tensor:
    if 'concat' in speech_feat:
        T = speech_feat['concat'].shape[0]
    else:
        T = speech_feat['visual'].shape[0]

    N = lip_feat['temporal'].shape[0]

    effective_frames = min(T, N * frames_per_clip)
    max_possible_frames = max(T, N * frames_per_clip)

    valid_ratio = effective_frames / max_possible_frames if max_possible_frames > 0 else 1.0
    valid_len = max(1, int(target_len * valid_ratio))

    mask = torch.zeros(target_len, dtype=torch.bool)
    mask[:valid_len] = True

    return mask


def batch_align_features(
    speech_feats: list,
    lip_feats: list,
    target_len: int = 25,
    use_concat: bool = True,
    create_mask: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    batch_size = len(speech_feats)
    sem_list = []
    dyn_list = []
    mask_list = []

    for i in range(batch_size):
        sem, dyn = align_temporal_features(
            speech_feats[i],
            lip_feats[i],
            target_len=target_len,
            use_concat=use_concat
        )
        sem_list.append(sem)
        dyn_list.append(dyn)

        if create_mask:
            mask = create_padding_mask(speech_feats[i], lip_feats[i], target_len)
            mask_list.append(mask)

    sem_batch = torch.stack(sem_list, dim=0)
    dyn_batch = torch.stack(dyn_list, dim=0)
    mask_batch = torch.stack(mask_list, dim=0) if create_mask else None

    return sem_batch, dyn_batch, mask_batch


def compute_temporal_statistics(
    speech_feat: Dict[str, torch.Tensor],
    lip_feat: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    T = speech_feat.get('concat', speech_feat['visual']).shape[0]
    N = lip_feat['temporal'].shape[0]

    temporal_ratio = T / (N * 25) if N > 0 else 0.0
    alignment_quality = 1.0 - abs(1.0 - temporal_ratio)

    return {
        'speech_frames': int(T),
        'lip_clips': int(N),
        'temporal_ratio': float(temporal_ratio),
        'alignment_quality': float(alignment_quality)
    }


def align_per_clip_features(
    speech_feat: Dict[str, torch.Tensor],
    lip_feat: Dict[str, torch.Tensor],
    target_len: int = 25,
    use_concat: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align features for per-clip prediction mode.

    Resamples each dynamic clip independently to target_len and replicates
    semantic features so every clip shares the full speech context.
    """
    if use_concat:
        if 'concat' in speech_feat:
            sem_raw = speech_feat['concat']
        else:
            sem_raw = torch.cat([speech_feat['visual'], speech_feat['audio']], dim=-1)
    else:
        sem_raw = speech_feat['visual']

    dyn_raw = lip_feat['temporal']

    # Detect old vs. new format
    if dyn_raw.ndim == 2:
        total_T = dyn_raw.shape[0]
        T_per_clip = lip_feat.get('frames_per_clip', 25)
        if total_T % T_per_clip == 0:
            num_clips = total_T // T_per_clip
            dyn_raw = dyn_raw.reshape(num_clips, T_per_clip, -1)
        else:
            raise ValueError(
                f"Cannot infer per-clip structure: total_T={total_T} "
                f"is not divisible by T_per_clip={T_per_clip}. "
                f"Re-extract features with DynamicForensics."
            )
    num_clips = dyn_raw.shape[0]

    sem_T = sem_raw.transpose(0, 1).unsqueeze(0)
    sem_pooled = F.adaptive_avg_pool1d(sem_T, output_size=target_len)
    sem_single = sem_pooled.squeeze(0).transpose(0, 1)

    dyn_clips = []
    for c in range(num_clips):
        clip = dyn_raw[c]
        clip_T = clip.transpose(0, 1).unsqueeze(0)
        clip_pooled = F.adaptive_avg_pool1d(clip_T, output_size=target_len)
        dyn_clips.append(clip_pooled.squeeze(0).transpose(0, 1))
    dyn_stacked = torch.stack(dyn_clips, dim=0)

    sem_stacked = sem_single.unsqueeze(0).expand(num_clips, -1, -1)

    return sem_stacked, dyn_stacked


if __name__ == '__main__':
    print("Testing feature alignment...")

    speech_feat = {
        'visual': torch.randn(100, 1024),
        'audio': torch.randn(100, 1024),
        'concat': torch.randn(100, 2048)
    }
    lip_feat = {
        'temporal': torch.randn(4, 512)
    }

    sem, dyn = align_temporal_features(speech_feat, lip_feat, target_len=64)
    print(f"Aligned shapes: sem={sem.shape}, dyn={dyn.shape}")

    mask = create_padding_mask(speech_feat, lip_feat, target_len=64)
    print(f"Mask shape: {mask.shape}, valid positions: {mask.sum().item()}/{len(mask)}")

    stats = compute_temporal_statistics(speech_feat, lip_feat)
    print(f"Temporal statistics: {stats}")

    batch_sem, batch_dyn, batch_mask = batch_align_features(
        [speech_feat, speech_feat],
        [lip_feat, lip_feat],
        target_len=64
    )
    print(f"Batch shapes: sem={batch_sem.shape}, dyn={batch_dyn.shape}, mask={batch_mask.shape}")
