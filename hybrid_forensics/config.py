import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._get_default_config()

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'speech_threshold': 0.275,
            'dynamic_threshold': -8.92,

            'crop_width': 96,
            'crop_height': 96,
            'window_margin': 12,
            'start_idx': 48,
            'stop_idx': 68,
            'std_size': (256, 256),
            'stable_points': [33, 36, 39, 42, 45],

            'speech_model_path': 'checkpoints/av_hubert/large_vox_iter5.pt',
            'dynamic_model_path': 'checkpoints/dynamic/dynamic_ff.pth',
            'mean_face_path': 'data/20words_mean_face.npy',

            'num_workers': 8,
            'max_frames_per_video': 110,
            'frames_per_clip': 25,
            'audio_sample_rate': 16000,
            'ffmpeg_path': '/usr/bin/ffmpeg',

            'video_root': None,
            'file_list': None,
            'speech_mouth_dir': None,
            'speech_audio_dir': None,
            'dynamic_mouth_dir': None,
            'output_dir': 'results/',

            'verbose': True,
            'save_detailed_results': True,
        }

    def _load_config(self, config_path: str) -> None:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f) or {}
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                loaded = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

        self.config.update(loaded)

    def update(self, **kwargs) -> None:
        self.config.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.config

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()

    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {output_path}")


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def resolve_path(path: str, relative_to_root: bool = True) -> str:
    if os.path.isabs(path):
        return path

    if relative_to_root:
        return str(get_project_root() / path)

    return os.path.abspath(path)


_global_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)

    return _global_config


def set_config(config: Config) -> None:
    global _global_config
    _global_config = config
