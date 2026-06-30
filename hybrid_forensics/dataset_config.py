import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class DatasetConfig:

    def __init__(self, config_path: Optional[str] = None):
        self.datasets = {}
        self.guidelines = {}
        self.metrics = {}

        if config_path is None:
            config_path = self._get_default_config_path()

        if os.path.exists(config_path):
            self._load_config(config_path)

    def _get_default_config_path(self) -> str:
        project_root = Path(__file__).parent.parent
        return str(project_root / 'configs' / 'dataset_thresholds.yaml')

    def _load_config(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config:
            self.datasets = config.get('datasets', {})
            self.guidelines = config.get('tuning_guidelines', {})
            self.metrics = config.get('performance_metrics', {})

    def get_dataset_names(self) -> List[str]:
        return list(self.datasets.keys())

    def get_thresholds(self, dataset_name: str) -> Dict[str, float]:
        if dataset_name not in self.datasets:
            available = ', '.join(self.get_dataset_names())
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {available}"
            )

        dataset_config = self.datasets[dataset_name]
        return {
            'speech_threshold': dataset_config.get('speech_threshold'),
            'dynamic_threshold': dataset_config.get('dynamic_threshold')
        }

    def get_speech_threshold(self, dataset_name: str) -> float:
        return self.get_thresholds(dataset_name)['speech_threshold']

    def get_dynamic_threshold(self, dataset_name: str) -> float:
        return self.get_thresholds(dataset_name)['dynamic_threshold']

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        return self.datasets[dataset_name]

    def get_metrics(self, dataset_name: str) -> Dict[str, float]:
        if dataset_name not in self.metrics:
            return {}

        return self.metrics[dataset_name]

    def print_all_datasets(self) -> None:
        print("\nAvailable datasets and thresholds:")
        print("-" * 40)

        for dataset_name, config in self.datasets.items():
            print(f"\n{dataset_name.upper()}")
            print(f"  Description: {config.get('description', 'N/A')}")
            print(f"  Speech Threshold: {config.get('speech_threshold')}")
            print(f"  Dynamic Threshold: {config.get('dynamic_threshold')}")
            print(f"  Notes: {config.get('notes', 'N/A')}")
            print(f"  Num Videos: {config.get('num_videos', 'N/A')}")

            if dataset_name in self.metrics:
                metrics = self.metrics[dataset_name]
                print(f"  Performance Metrics:")
                print(f"    AUC: {metrics.get('auc', 'N/A')}")
                print(f"    Accuracy: {metrics.get('accuracy', 'N/A')}")
                print(f"    Precision: {metrics.get('precision', 'N/A')}")
                print(f"    Recall: {metrics.get('recall', 'N/A')}")
                print(f"    F1: {metrics.get('f1', 'N/A')}")

        print()

    def print_guidelines(self) -> None:
        print("\nThreshold tuning guidelines:")
        print("-" * 40)

        for threshold_name, guideline in self.guidelines.items():
            print(f"\n{threshold_name.upper()}")
            print(f"  Description: {guideline.get('description', 'N/A')}")
            print(f"  Range: {guideline.get('range', 'N/A')}")
            print(f"  Interpretation: {guideline.get('interpretation', 'N/A')}")
            print(f"  Lower Value: {guideline.get('lower_value', 'N/A')}")
            print(f"  Higher Value: {guideline.get('higher_value', 'N/A')}")

        print()


_global_dataset_config = None


def get_dataset_config(config_path: Optional[str] = None) -> DatasetConfig:
    global _global_dataset_config

    if _global_dataset_config is None:
        _global_dataset_config = DatasetConfig(config_path)

    return _global_dataset_config


def get_dataset_thresholds(dataset_name: str) -> Dict[str, float]:
    config = get_dataset_config()
    return config.get_thresholds(dataset_name)


if __name__ == '__main__':
    config = DatasetConfig()

    print("Available datasets:", config.get_dataset_names())

    config.print_all_datasets()

    config.print_guidelines()

    print("\nLRS2 Thresholds:", config.get_thresholds('lrs2'))
    print("LRW Thresholds:", config.get_thresholds('lrw'))
    print("FakeAVCeleb Thresholds:", config.get_thresholds('fakeavceleb'))
    print("AVLips Thresholds:", config.get_thresholds('avlips'))
