import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_forensics.models.model_loader import load_speech_model, load_dynamic_model
from hybrid_forensics.inference.speech_forensics import SpeechForensics
from hybrid_forensics.inference.dynamic_forensics import DynamicForensics


def extract_features(
    video_list_file: str,
    speech_mouth_dir: str,
    speech_audio_dir: str,
    dynamic_mouth_dir: str,
    output_dir: str,
    checkpoint_path: str = "checkpoints/av_hubert/large_vox_iter5.pt",
    dynamic_model_path: str = "checkpoints/dynamic/dynamic_ff.pth",
    max_length: int = 10,
    skip_existing: bool = True,
    frames_per_clip: int = 25,
) -> None:
    print("=" * 60)
    print("TriCon Feature Extraction")
    print("=" * 60)

    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("  Loading SpeechForensics (AV-HuBERT)...")
    speech_model, speech_task = load_speech_model(checkpoint_path)
    speech_detector = SpeechForensics(speech_model, speech_task.cfg, max_length=max_length)

    print(f"  Loading DynamicForensics (MS-TCN, frames_per_clip={frames_per_clip})...")
    dynamic_model, dynamic_transform = load_dynamic_model(dynamic_model_path)
    dynamic_detector = DynamicForensics(dynamic_model, dynamic_transform, frames_per_clip=frames_per_clip)

    print("Models loaded successfully")

    print(f"\nReading video list from {video_list_file}...")
    with open(video_list_file, 'r', encoding='utf-8') as f:
        video_list = [line.strip() for line in f if line.strip()]

    print(f"Found {len(video_list)} videos")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExtracting features to {output_dir}...")
    success_count = 0
    skip_count = 0
    error_count = 0
    error_log = []

    for video_item in tqdm(video_list, desc="Processing videos"):
        parts = video_item.rsplit(maxsplit=1) if ' ' in video_item else video_item.rsplit(',', maxsplit=1)
        if len(parts) < 2:
            error_log.append(f"Invalid format: {video_item}")
            error_count += 1
            continue

        video_rel_path = parts[0]
        video_label = int(parts[1])
        video_base = os.path.splitext(video_rel_path)[0]

        speech_mouth = os.path.join(speech_mouth_dir, video_base, 'speech_mouth.mp4')
        speech_audio = os.path.join(speech_audio_dir, video_base + '.wav')
        dynamic_mouth = os.path.join(dynamic_mouth_dir, video_base, 'lip_mouth')

        feat_dir = os.path.join(output_dir, video_base)
        os.makedirs(feat_dir, exist_ok=True)

        speech_feat_path = os.path.join(feat_dir, 'speech_features.pt')
        dynamic_feat_path = os.path.join(feat_dir, 'dynamic_features.pt')

        if skip_existing and os.path.exists(speech_feat_path) and os.path.exists(dynamic_feat_path):
            skip_count += 1
            continue

        try:
            if not os.path.exists(dynamic_mouth):
                error_log.append(f"{video_base}: Missing dynamic_mouth: {dynamic_mouth}")
                error_count += 1
                continue

            logit, dynamic_feat = dynamic_detector.detect(dynamic_mouth, return_features=True)
            dyn_temporal = dynamic_feat['temporal']

            torch.save({
                'temporal': dyn_temporal,
                'avg_logit': logit,
                'label': video_label,
            }, dynamic_feat_path)

            if os.path.exists(speech_mouth) and os.path.exists(speech_audio):
                similarity, speech_feat = speech_detector.detect(
                    speech_mouth, speech_audio, return_features=True
                )
                torch.save({
                    'visual': speech_feat['visual'],
                    'audio': speech_feat['audio'],
                    'concat': speech_feat['concat'],
                    'similarity': similarity,
                    'label': video_label,
                }, speech_feat_path)

            else:
                missing = []
                if not os.path.exists(speech_mouth):
                    missing.append(f"speech_mouth: {speech_mouth}")
                if not os.path.exists(speech_audio):
                    missing.append(f"audio: {speech_audio}")
                error_log.append(f"{video_base}: Missing {', '.join(missing)}")
                error_count += 1
                continue

            success_count += 1

        except Exception as e:
            error_log.append(f"{video_base}: {str(e)}")
            error_count += 1
            continue

    print("\nExtraction complete!")
    print("=" * 60)
    print(f"\nDone.  Success: {success_count}  |  Skipped: {skip_count}  |  Errors: {error_count}")

    if error_log:
        error_log_path = os.path.join(output_dir, "extraction_errors.log")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(error_log))
        print(f"\nError log saved to {error_log_path}")

    print(f"\nFeatures saved to {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache features for TriCon training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--video_list',
        required=True,
        help='Path to file_list.txt (format: "video_path label")'
    )
    parser.add_argument(
        '--speech_mouth_dir',
        required=True,
        help='Directory containing speech_mouth.mp4 files'
    )
    parser.add_argument(
        '--speech_audio_dir',
        required=True,
        help='Directory containing audio .wav files'
    )
    parser.add_argument(
        '--dynamic_mouth_dir',
        required=True,
        help='Directory containing dynamic_mouth/*.png sequences'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for cached features'
    )
    parser.add_argument(
        '--checkpoint_path',
        default='checkpoints/av_hubert/large_vox_iter5.pt',
        help='Path to AV-HuBERT checkpoint'
    )
    parser.add_argument(
        '--dynamic_model_path',
        default='checkpoints/dynamic/dynamic_ff.pth',
        help='Path to DynamicForensics checkpoint'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=10,
        help='Maximum video length in seconds for SpeechForensics'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=True,
        help='Skip videos that already have cached features'
    )
    parser.add_argument(
        '--frames_per_clip',
        type=int,
        default=25,
        help='Number of frames per clip for DynamicForensics MS-TCN. '
             'Determines temporal granularity of dynamic features. '
             '(default: 25)'
    )

    args = parser.parse_args()

    extract_features(
        video_list_file=args.video_list,
        speech_mouth_dir=args.speech_mouth_dir,
        speech_audio_dir=args.speech_audio_dir,
        dynamic_mouth_dir=args.dynamic_mouth_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        dynamic_model_path=args.dynamic_model_path,
        max_length=args.max_length,
        skip_existing=args.skip_existing,
        frames_per_clip=args.frames_per_clip,
    )


if __name__ == '__main__':
    main()
