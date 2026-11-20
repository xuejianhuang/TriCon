"""
Command-line interface for TriCon
"""

import argparse
from hybrid_forensics.preprocessing.extract_landmarks import extract_landmarks_main
from hybrid_forensics.preprocessing.crop_mouths import crop_mouths_main
from hybrid_forensics.preprocessing.extract_audio import extract_audio_main


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TriCon: Deepfake Detection Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract landmarks
  python -m hybrid_forensics preprocess landmarks --video_root data/videos --file_list data/list.txt --output_dir data/landmarks

  # Crop mouths
  python -m hybrid_forensics preprocess crop --video_root data/videos --landmarks_dir data/landmarks --output_dir data/mouths

  # Extract audio
  python -m hybrid_forensics preprocess audio --video_root data/videos --file_list data/list.txt --output_dir data/audio
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Preprocessing subcommand
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocessing commands')
    preprocess_subparsers = preprocess_parser.add_subparsers(dest='preprocess_cmd', help='Preprocessing step')

    # Landmarks extraction
    landmarks_parser = preprocess_subparsers.add_parser('landmarks', help='Extract facial landmarks')
    landmarks_parser.add_argument('--video_root', type=str, required=True, help='Root directory of videos')
    landmarks_parser.add_argument('--file_list', type=str, required=True, help='Path to file list')
    landmarks_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    landmarks_parser.add_argument('--face_detector', type=str, default='checkpoints/Resnet50_Final.pth')
    landmarks_parser.add_argument('--max_frames', type=int, default=None)

    # Mouth cropping
    crop_parser = preprocess_subparsers.add_parser('crop', help='Crop mouth regions')
    crop_parser.add_argument('--video_root', type=str, required=True)
    crop_parser.add_argument('--landmarks_dir', type=str, required=True)
    crop_parser.add_argument('--output_dir', type=str, required=True)
    crop_parser.add_argument('--mean_face', type=str, default='data/20words_mean_face.npy')
    crop_parser.add_argument('--crop_width', type=int, default=96)
    crop_parser.add_argument('--crop_height', type=int, default=96)
    crop_parser.add_argument('--start_idx', type=int, default=48)
    crop_parser.add_argument('--stop_idx', type=int, default=68)
    crop_parser.add_argument('--window_margin', type=int, default=12)
    crop_parser.add_argument('--num_workers', type=int, default=8)
    crop_parser.add_argument('--skip_existing', action='store_true')

    # Audio extraction
    audio_parser = preprocess_subparsers.add_parser('audio', help='Extract audio')
    audio_parser.add_argument('--video_root', type=str, required=True)
    audio_parser.add_argument('--file_list', type=str, required=True)
    audio_parser.add_argument('--output_dir', type=str, required=True)
    audio_parser.add_argument('--ffmpeg', type=str, default='/usr/bin/ffmpeg')
    audio_parser.add_argument('--sample_rate', type=int, default=16000)

    args = parser.parse_args()

    if args.command == 'preprocess':
        if args.preprocess_cmd == 'landmarks':
            extract_landmarks_main(args)
        elif args.preprocess_cmd == 'crop':
            crop_mouths_main(args)
        elif args.preprocess_cmd == 'audio':
            extract_audio_main(args)
        else:
            preprocess_parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
