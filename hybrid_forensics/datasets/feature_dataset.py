import os
import os.path as osp
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    return tensor.float()


def _resample_sequence(sequence: torch.Tensor, target_len: Optional[int]) -> torch.Tensor:
    if target_len is None or sequence.size(0) == target_len:
        return sequence
    if sequence.size(0) == 1:
        return sequence.repeat(target_len, 1)
    seq = sequence.unsqueeze(0).transpose(1, 2)
    seq = F.interpolate(seq, size=target_len, mode="linear", align_corners=False)
    seq = seq.transpose(1, 2).squeeze(0)
    return seq


def _resample_per_clip(temporal: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample each clip's features independently to target_len. Returns [N, target_len, D]."""
    N = temporal.shape[0]
    resampled = []
    for i in range(N):
        resampled.append(_resample_sequence(temporal[i], target_len))
    return torch.stack(resampled, dim=0)


class FeatureDataset(Dataset):

    def __init__(
        self,
        file_list: str,
        feature_root: str,
        seq_len: int = 25,
        require_dynamic: bool = True,
        include_audio: bool = False,
        use_concat: bool = True,
        create_mask: bool = True,
        augment: bool = False,
        use_frequent: bool = False,
        frequent_root: str = None,
        real_only: bool = False,
        sem_truncate: int = 0,
        dyn_truncate: int = 0,
        real_file_list: str = None,
        use_per_clip: bool = True,
    ) -> None:
        super().__init__()
        self.feature_root = feature_root
        self.seq_len = seq_len
        self.require_dynamic = require_dynamic
        self.include_audio = include_audio
        self.use_concat = use_concat
        self.create_mask = create_mask
        self.augment = augment
        self.use_frequent = use_frequent
        self.sem_truncate = sem_truncate
        self.dyn_truncate = dyn_truncate
        self.use_per_clip = use_per_clip

        if use_frequent:
            if frequent_root is not None:
                self.frequent_root = frequent_root
            else:
                self.frequent_root = feature_root.replace(
                    "cached_features", "frequent_features"
                )

        self.samples = []
        self._load_file_list(file_list)
        if real_file_list is not None:
            self._load_file_list(real_file_list)

        if real_only:
            self.samples = [s for s in self.samples if s["label"] == 0]

        if len(self.samples) == 0:
            raise RuntimeError("No samples found for FeatureDataset. Check feature paths and file list.")

    def _load_file_list(self, file_list: str) -> None:
        with open(file_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(None, 1) if ' ' in line else line.rsplit(',', 1)
                if len(parts) < 2:
                    continue
                rel_path, label_str = parts
                rel_no_ext = osp.splitext(rel_path)[0]
                sample_dir = osp.join(self.feature_root, rel_no_ext)
                speech_path = osp.join(sample_dir, "speech_features.pt")
                dynamic_path = osp.join(sample_dir, "dynamic_features.pt")

                if not osp.exists(speech_path):
                    continue
                if self.require_dynamic and not osp.exists(dynamic_path):
                    continue

                frequent_path = None
                if self.use_frequent:
                    freq_dir = osp.join(self.frequent_root, rel_no_ext)
                    frequent_path = osp.join(freq_dir, "frequent_features.pt")
                    if not osp.exists(frequent_path):
                        continue

                self.samples.append(
                    {
                        "video": rel_path,
                        "label": int(label_str),
                        "speech_path": speech_path,
                        "dynamic_path": dynamic_path if osp.exists(dynamic_path) else None,
                        "frequent_path": frequent_path,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        try:
            sample = self.samples[idx]
            speech_data = torch.load(sample["speech_path"], map_location="cpu")

            if self.use_concat and "concat" in speech_data:
                sem_feat = _to_tensor(speech_data["concat"])
            elif self.use_concat:
                visual = _to_tensor(speech_data.get("visual"))
                audio = _to_tensor(speech_data.get("audio"))
                if visual.shape[0] != audio.shape[0]:
                    audio = _resample_sequence(audio, visual.shape[0])
                sem_feat = torch.cat([visual, audio], dim=-1)
            else:
                sem_feat = _to_tensor(speech_data.get("visual"))

            sem_feat_original_len = sem_feat.shape[0]
            sem_feat = _resample_sequence(sem_feat, self.seq_len)

            if self.sem_truncate > 0:
                sem_feat = sem_feat[:, :self.sem_truncate]

            dynamic_data = None
            dyn_feat = None
            dyn_feat_original_len = 0
            num_clips = 1
            if sample["dynamic_path"] and osp.exists(sample["dynamic_path"]):
                dynamic_data = torch.load(sample["dynamic_path"], map_location="cpu")
                temporal = dynamic_data.get("temporal")
                if temporal is not None:
                    temporal_t = _to_tensor(temporal)

                    if temporal_t.dim() == 3:
                        # [N, T_per_clip, D] — native per-clip format
                        num_clips = temporal_t.shape[0]
                        dyn_feat_original_len = num_clips * temporal_t.shape[1]
                        if self.use_per_clip:
                            dyn_feat = _resample_per_clip(temporal_t, self.seq_len)
                        else:
                            flat = temporal_t.reshape(-1, temporal_t.shape[-1])
                            dyn_feat_original_len = flat.shape[0]
                            dyn_feat = _resample_sequence(flat, self.seq_len)
                    elif temporal_t.dim() == 2:
                        # [total_T, D] — legacy concatenated format
                        total_T = temporal_t.shape[0]
                        if self.use_per_clip:
                            T_per_clip = dynamic_data.get("frames_per_clip", 25)
                            if total_T % T_per_clip == 0:
                                num_clips = total_T // T_per_clip
                                temporal_3d = temporal_t.reshape(num_clips, T_per_clip, -1)
                                dyn_feat_original_len = total_T
                                dyn_feat = _resample_per_clip(temporal_3d, self.seq_len)
                            else:
                                dyn_feat_original_len = total_T
                                dyn_feat = _resample_sequence(temporal_t, self.seq_len)
                                dyn_feat = dyn_feat.unsqueeze(0)
                        else:
                            dyn_feat_original_len = total_T
                            dyn_feat = _resample_sequence(temporal_t, self.seq_len)
                    else:
                        dyn_feat_original_len = temporal_t.shape[0]
                        dyn_feat = _resample_sequence(temporal_t, self.seq_len)

            if dyn_feat is None:
                if self.require_dynamic:
                    raise RuntimeError(f"Dynamic features missing for {sample['video']}")
                dyn_feat = torch.zeros(self.seq_len, 768)

            if self.dyn_truncate > 0:
                if dyn_feat.dim() == 3:
                    dyn_feat = dyn_feat[:, :, :self.dyn_truncate]
                else:
                    dyn_feat = dyn_feat[:, :self.dyn_truncate]

            # replicate semantic features to match num_clips
            if self.use_per_clip and dyn_feat.dim() == 3:
                num_clips = dyn_feat.shape[0]
                sem_feat = sem_feat.unsqueeze(0).expand(num_clips, -1, -1)
                # sem_feat: [num_clips, seq_len, sem_dim]

            freq_feat = None
            if self.use_frequent and sample.get("frequent_path"):
                if osp.exists(sample["frequent_path"]):
                    freq_data = torch.load(sample["frequent_path"], map_location="cpu")
                    freq_temporal = freq_data.get("frequent")
                    if freq_temporal is not None:
                        freq_feat = _resample_sequence(_to_tensor(freq_temporal), self.seq_len)
                if freq_feat is None:
                    freq_feat = torch.zeros(self.seq_len, 768)

            if self.dyn_truncate > 0 and freq_feat is not None:
                if freq_feat.dim() == 3:
                    freq_feat = freq_feat[:, :, :self.dyn_truncate]
                else:
                    freq_feat = freq_feat[:, :self.dyn_truncate]

            audio_feat = None
            if self.include_audio and speech_data.get("audio") is not None:
                audio_feat = _resample_sequence(_to_tensor(speech_data["audio"]), self.seq_len)

            mask = None
            if self.create_mask and not self.use_per_clip and dyn_feat_original_len > 0:
                mask = self._create_mask(sem_feat_original_len, dyn_feat_original_len)

            result = {
                "video": sample["video"],
                "label": torch.tensor(sample["label"], dtype=torch.long),
                "sem_feat": sem_feat,
                "dyn_feat": dyn_feat,
                "speech_similarity": float(speech_data.get("similarity", 0.0)),
                "dynamic_logit": float(dynamic_data.get("avg_logit", 0.0)) if dynamic_data else 0.0,
            }

            if audio_feat is not None:
                result["audio_feat"] = audio_feat

            if freq_feat is not None:
                result["freq_feat"] = freq_feat

            if mask is not None:
                result["mask"] = mask

            if self.augment:
                result = self._augment_features(result)

            return result

        except Exception as e:
            print(f"Error loading sample {idx} (video: {sample.get('video', 'unknown')}): {e}")
            raise

    def _create_mask(self, sem_len: int, dyn_len: int) -> torch.Tensor:
        effective_frames = min(sem_len, dyn_len * 25)
        max_possible_frames = max(sem_len, dyn_len * 25)

        valid_ratio = effective_frames / max_possible_frames if max_possible_frames > 0 else 1.0
        valid_len = max(1, int(self.seq_len * valid_ratio))

        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[:valid_len] = True

        return mask

    def _augment_features(self, sample: Dict) -> Dict:
        # dim=-2 works for both 2-D [T,D] and 3-D [N,T,D] tensors

        if torch.rand(1) < 0.3:
            sample["sem_feat"] = torch.flip(sample["sem_feat"], dims=[-2])
            sample["dyn_feat"] = torch.flip(sample["dyn_feat"], dims=[-2])
            if sample.get("freq_feat") is not None:
                sample["freq_feat"] = torch.flip(sample["freq_feat"], dims=[-2])

        if torch.rand(1) < 0.3:
            shift = torch.randint(-self.seq_len // 4, self.seq_len // 4, (1,)).item()
            sample["sem_feat"] = torch.roll(sample["sem_feat"], shifts=shift, dims=-2)
            sample["dyn_feat"] = torch.roll(sample["dyn_feat"], shifts=shift, dims=-2)
            if sample.get("freq_feat") is not None:
                sample["freq_feat"] = torch.roll(sample["freq_feat"], shifts=shift, dims=-2)

        if torch.rand(1) < 0.2:
            dropout_prob = 0.1
            sem_mask = torch.rand_like(sample["sem_feat"]) > dropout_prob
            sample["sem_feat"] = sample["sem_feat"] * sem_mask

            dyn_mask = torch.rand_like(sample["dyn_feat"]) > dropout_prob
            sample["dyn_feat"] = sample["dyn_feat"] * dyn_mask

            if sample.get("freq_feat") is not None:
                freq_mask = torch.rand_like(sample["freq_feat"]) > dropout_prob
                sample["freq_feat"] = sample["freq_feat"] * freq_mask

        return sample


def _safe_collate(batch):
    from torch.utils.data.dataloader import default_collate

    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    return default_collate(batch)


def _per_clip_collate(batch):
    """Collate for per-clip mode: flatten all clips, track video membership via sample_ids."""
    from torch.utils.data.dataloader import default_collate

    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # If any sample is NOT per-clip (2-D dyn_feat), fall back to safe collate
    if any(item["dyn_feat"].dim() == 2 for item in batch):
        return _safe_collate(batch)

    all_sem = []
    all_dyn = []
    all_labels = []
    all_freq = []
    sample_ids = []
    videos = []
    speech_sims = []
    dynamic_logits = []

    for idx, item in enumerate(batch):
        n_clips = item["dyn_feat"].shape[0]
        all_sem.append(item["sem_feat"])
        all_dyn.append(item["dyn_feat"])
        all_labels.append(item["label"].repeat(n_clips))
        sample_ids.append(torch.full((n_clips,), idx, dtype=torch.long))

        if item.get("freq_feat") is not None:
            all_freq.append(item["freq_feat"])

        videos.append(item["video"])
        speech_sims.append(item.get("speech_similarity", 0.0))
        dynamic_logits.append(item.get("dynamic_logit", 0.0))

    result = {
        "sem_feat": torch.cat(all_sem, dim=0),
        "dyn_feat": torch.cat(all_dyn, dim=0),
        "label": torch.cat(all_labels, dim=0),
        "sample_ids": torch.cat(sample_ids, dim=0),
        "num_samples": len(batch),
        "video": videos,
        "speech_similarity": torch.tensor(speech_sims),
        "dynamic_logit": torch.tensor(dynamic_logits),
    }

    if all_freq:
        result["freq_feat"] = torch.cat(all_freq, dim=0)

    return result


def create_dataloader(
    file_list: str,
    feature_root: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    real_only: bool = False,
    real_file_list: str = None,
    use_per_clip: bool = True,
    **dataset_kwargs,
) -> Tuple[FeatureDataset, DataLoader]:
    dataset = FeatureDataset(
        file_list=file_list,
        feature_root=feature_root,
        real_only=real_only,
        real_file_list=real_file_list,
        use_per_clip=use_per_clip,
        **dataset_kwargs,
    )
    collate_fn = _per_clip_collate if use_per_clip else _safe_collate
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return dataset, loader
