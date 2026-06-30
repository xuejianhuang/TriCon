#!/usr/bin/env python3
"""
Landmark extraction script for CSV-based datasets.
Reads video information from a metadata CSV and extracts 68 facial landmarks.
"""

import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
import os.path as osp
import sys
from skimage import io
import face_alignment
from glob import glob
import json
import argparse
import pandas as pd
from pathlib import Path

def detect_save_landmark_68_csv(args):
    """Read video info from CSV and extract facial landmarks."""
    csv_path = args.csv_path
    video_root = args.video_root
    out_dir = args.out_dir

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    if 'category' in df.columns:
        print(f"Category distribution: {df['category'].value_counts().to_dict()}")

    os.makedirs(out_dir, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    print("Starting landmark extraction...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        if args.use_full_paths:
            video_path = row['original_path']
        else:
            video_path = osp.join(video_root, row['original_path'])

        video_id = row['video_id']
        segment_id = row['segment_id']
        out_path = osp.join(out_dir, f"{video_id}_{segment_id}.json")

        if osp.exists(out_path) and not args.force:
            skipped_count += 1
            continue

        if not osp.exists(video_path):
            print(f"Video not found: {video_path}")
            failed_count += 1
            continue

        try:
            os.makedirs(osp.dirname(out_path), exist_ok=True)

            frames = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                failed_count += 1
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames) == 0:
                print(f"Empty video: {video_path}")
                failed_count += 1
                continue

            frames = np.asarray(frames)

            landmarks = {}
            for i in range(len(frames)):
                frame = frames[i]
                landmark = fa.get_landmarks(frame)
                if (landmark is not None) and (len(landmark) != 0):
                    landmark = landmark[0]
                    landmark = landmark.tolist()
                else:
                    landmark = None

                img_name = f'{i:04d}.jpg'
                landmarks[img_name] = landmark

            with open(out_path, 'w') as f:
                json.dump(landmarks, f, indent=2)

            processed_count += 1

        except Exception as e:
            print(f"Processing failed {video_path}: {e}")
            failed_count += 1
            continue

    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed_count} videos")
    print(f"  Skipped: {skipped_count} videos")
    print(f"  Failed: {failed_count} videos")
    print(f"  Output directory: {out_dir}")

def create_file_list_from_csv(csv_path, output_file, video_root="", use_full_paths=False):
    """Create a file list from a CSV dataset."""
    df = pd.read_csv(csv_path)

    with open(output_file, 'w') as f:
        for idx, row in df.iterrows():
            if use_full_paths:
                video_path = row['original_path']
            else:
                video_path = osp.join(video_root, row['original_path'])

            f.write(f"{video_path} {row['label']}\n")

    print(f"File list saved to: {output_file}")
    print(f"Contains {len(df)} videos")

def main():
    parser = argparse.ArgumentParser(
        description='Landmark extraction script for CSV-based datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV dataset file')
    parser.add_argument('--video_root', type=str, default='',
                       help='Root directory for videos (if not using full paths)')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for landmarks')
    parser.add_argument('--use_full_paths', action='store_true',
                       help='Use full paths from CSV instead of relative paths')

    parser.add_argument('--face_detector', type=str,
                       default='checkpoints/Resnet50_Final.pth',
                       help='Path to face detector model')
    parser.add_argument('--face_predictor', type=str,
                       default='checkpoints/2DFAN4-cd938726ad.zip',
                       help='Path to landmark predictor model')

    parser.add_argument('--force', action='store_true',
                       help='Force re-processing of existing files')
    parser.add_argument('--create_file_list', type=str, default='',
                       help='Create file list and save to specified path')
    parser.add_argument('--ffmpeg', type=str, default='/usr/bin/ffmpeg',
                       help='Path to ffmpeg executable')

    args = parser.parse_args()

    if not osp.exists(args.csv_path):
        print(f"CSV file not found: {args.csv_path}")
        return

    if not osp.exists(args.face_detector):
        print(f"Face detector not found: {args.face_detector}")
        print("Please download the RetinaFace model to the checkpoints/ directory")
        return

    if args.create_file_list:
        create_file_list_from_csv(
            args.csv_path,
            args.create_file_list,
            args.video_root,
            args.use_full_paths
        )
        return

    print("Initializing Face Alignment...")
    try:
        global fa
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector='retinaface',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            face_detector_kwargs={'path_to_detector': args.face_detector}
        )
        print("Face Alignment initialized successfully")
    except Exception as e:
        print(f"Face Alignment initialization failed: {e}")
        return

    detect_save_landmark_68_csv(args)

if __name__ == '__main__':
    main()
