import os
import torch
import numpy as np
from PIL import Image
import os.path as osp


class LipForensics:

    def __init__(self, model, transform, frames_per_clip=25, max_frames=110):
        self.model = model
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.max_frames = max_frames

    def load_frames(self, frames_dir):
        if not osp.exists(frames_dir):
            return None

        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if len(frames) == 0:
            return None

        frames = frames[:self.max_frames]

        sample = []
        for frame_file in frames:
            frame_path = osp.join(frames_dir, frame_file)
            try:
                with Image.open(frame_path) as pil_img:
                    pil_img = pil_img.convert("L")
                    img = np.array(pil_img)
                sample.append(img)
            except Exception as e:
                continue

        if len(sample) == 0:
            return None

        sample = np.stack(sample)
        return sample

    def preprocess_clip(self, clip_frames):
        clip_tensor = torch.from_numpy(clip_frames).unsqueeze(-1)

        if self.transform is not None:
            clip_tensor = self.transform(clip_tensor)

        clip_tensor = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        return clip_tensor

    def detect(self, lip_mouth_dir, return_features=False):
        try:
            frames = self.load_frames(lip_mouth_dir)
            if frames is None:
                return None

            num_frames = len(frames)
            num_clips = num_frames // self.frames_per_clip

            if num_clips == 0:
                return None

            logits = []
            temporal_features = []

            for clip_idx in range(num_clips):
                start_idx = clip_idx * self.frames_per_clip
                end_idx = start_idx + self.frames_per_clip

                clip_frames = frames[start_idx:end_idx]
                clip_tensor = self.preprocess_clip(clip_frames)
                clip_tensor = clip_tensor.cuda()

                with torch.no_grad():
                    if return_features:
                        logit, t_features = self.model(
                            clip_tensor,
                            lengths=[self.frames_per_clip],
                            return_features=True
                        )
                        temporal_features.append(t_features.squeeze(0).cpu())
                    else:
                        logit = self.model(
                            clip_tensor,
                            lengths=[self.frames_per_clip],
                            return_features=False
                        )
                    logits.append(logit.squeeze().cpu().item())

            avg_logit = float(np.mean(logits))

            if return_features:
                feature_dict = {
                    'temporal': torch.stack(temporal_features, dim=0) if temporal_features else None,
                    'frames_per_clip': self.frames_per_clip,
                    'per_clip': True,
                }
                return avg_logit, feature_dict

            return avg_logit

        except Exception as e:
            print(f"  LipForensics error: {e}")
            return None

    def detect_clip_level(self, lip_mouth_dir):
        try:
            frames = self.load_frames(lip_mouth_dir)
            if frames is None:
                return None

            num_frames = len(frames)
            num_clips = num_frames // self.frames_per_clip

            if num_clips == 0:
                return None

            clip_logits = []

            for clip_idx in range(num_clips):
                start_idx = clip_idx * self.frames_per_clip
                end_idx = start_idx + self.frames_per_clip

                clip_frames = frames[start_idx:end_idx]
                clip_tensor = self.preprocess_clip(clip_frames)
                clip_tensor = clip_tensor.cuda()

                with torch.no_grad():
                    logit = self.model(
                        clip_tensor,
                        lengths=[self.frames_per_clip],
                        return_features=False
                    )
                    clip_logits.append(logit.squeeze().cpu().item())

            return clip_logits

        except Exception as e:
            print(f"  LipForensics clip-level error: {e}")
            return None
