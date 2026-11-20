import os
import os.path as osp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import torch
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
from multiprocessing import Pool

def setup_face_alignment_path():
    face_alignment_paths = [
        os.path.join(os.path.dirname(__file__), 'face-alignment'),

    ]

    for path in face_alignment_paths:
        if os.path.exists(path):
            abs_path = os.path.abspath(path)
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            return True

    return False

setup_face_alignment_path()

try:
    import face_alignment
    HAS_FACE_ALIGNMENT = True
except ImportError:
    HAS_FACE_ALIGNMENT = False


def extract_landmarks_from_video(video_path, fa, max_frames=None):
    landmarks_dict = {}

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            landmark = fa.get_landmarks(frame_rgb)
            if landmark is not None and len(landmark) > 0:
                landmark = landmark[0].tolist()
            else:
                landmark = None
        except Exception as e:
            landmark = None

        frame_name = f'{frame_idx:04d}.jpg'
        landmarks_dict[frame_name] = landmark
        frame_idx += 1

    cap.release()
    return landmarks_dict


def process_video_wrapper(args_tuple):
    video_rel_path, video_root, output_dir, face_detector_path, max_frames, device_type = args_tuple

    video_path = osp.join(video_root, video_rel_path)
    video_rel_path_no_ext = osp.splitext(video_rel_path)[0]
    out_path = osp.join(output_dir, video_rel_path_no_ext, 'landmarks.json')

    if osp.exists(out_path):
        return ('already_processed', video_rel_path)

    if not osp.exists(video_path):
        return ('skipped', video_rel_path)

    try:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector='retinaface',
            device=device_type,
            face_detector_kwargs={'path_to_detector': face_detector_path}
        )

        landmarks_dict = extract_landmarks_from_video(video_path, fa, max_frames=max_frames)

        os.makedirs(osp.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(landmarks_dict, f)

        return ('processed', video_rel_path)
    except Exception as e:
        return ('error', (video_rel_path, str(e)))


def extract_landmarks_main(args):
    print("="*60)
    print("Extracting Facial Landmarks")
    print("="*60)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    from ..config import resolve_path
    face_detector_path = resolve_path(args.face_detector)

    print(f"Using face detector: {face_detector_path}")

    with open(args.file_list, 'r', encoding='utf-8') as f:
        video_list = f.read().split('\n')

    video_list = [v.strip() for v in video_list if v.strip()]
    print(f"\nFound {len(video_list)} videos to process")

    num_workers = getattr(args, 'num_workers', 1)
    print(f"Using {num_workers} worker(s)")

    if num_workers > 1:
        from functools import partial

        worker_args = [
            (video_item.split('\t')[0], args.video_root, args.output_dir,
             face_detector_path, args.max_frames, device)
            for video_item in video_list
        ]

        processed = 0
        skipped = 0
        errors = 0

        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_video_wrapper, worker_args),
                total=len(worker_args),
                desc="Extracting landmarks"
            ))

        for result in results:
            status = result[0]
            if status == 'processed':
                processed += 1
            elif status == 'already_processed':
                processed += 1
            elif status == 'skipped':
                skipped += 1
            else:
                errors += 1

        print("\n" + "="*60)
        print("Landmark Extraction Complete")
        print("="*60)
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Errors: {errors}")
        print(f"Output directory: {args.output_dir}")
    else:
        print("\nInitializing face alignment detector...")

        try:
            print("Loading RetinaFace detector...")
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                face_detector='retinaface',
                device=device,
                face_detector_kwargs={'path_to_detector': face_detector_path}
            )
            print("Face detector (RetinaFace) loaded successfully!")
        except Exception as e:
            print(f"\nFailed to load RetinaFace: {e}")
            raise RuntimeError(f"Could not load face detector: {e}")

        processed = 0
        skipped = 0

        for video_item in tqdm(video_list, desc="Extracting landmarks"):
            video_rel_path = video_item.split('\t')[0]
            video_path = osp.join(args.video_root, video_rel_path)

            if not osp.exists(video_path):
                skipped += 1
                continue

            video_rel_path_no_ext = osp.splitext(video_rel_path)[0]
            out_path = osp.join(args.output_dir, video_rel_path_no_ext, 'landmarks.json')

            if osp.exists(out_path):
                processed += 1
                continue

            try:
                landmarks_dict = extract_landmarks_from_video(
                    video_path, fa, max_frames=args.max_frames
                )

                os.makedirs(osp.dirname(out_path), exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump(landmarks_dict, f)

                processed += 1
            except Exception as e:
                print(f"  Error processing {video_path}: {e}")
                skipped += 1

        print("\n" + "="*60)
        print("Landmark Extraction Complete")
        print("="*60)
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract facial landmarks from videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--video_root',
        type=str,
        required=True,
        help='Root directory of videos'
    )
    parser.add_argument(
        '--file_list',
        type=str,
        required=True,
        help='Path to file list (format: "video_path label")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for landmarks'
    )
    parser.add_argument(
        '--face_detector',
        type=str,
        default='checkpoints/Resnet50_Final.pth',
        help='Path to face detector model'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Maximum frames per video (None for all)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )

    args = parser.parse_args()
    extract_landmarks_main(args)
