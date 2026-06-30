import os
import os.path as osp
import json
import numpy as np
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import cv2

from .utils import warp_img, apply_transform, cut_patch, landmarks_interpolate


STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]


def crop_video_and_save(video_path, landmarks_dir, target_dir, mean_face_landmarks, args, skip_existing=False):
    if skip_existing and osp.exists(target_dir):
        mp4_path = osp.join(target_dir, 'speech_mouth.mp4')
        png_dir = osp.join(target_dir, 'lip_mouth')
        if osp.exists(mp4_path) and osp.exists(png_dir):
            existing_pngs = [f for f in os.listdir(png_dir) if f.endswith('.png')]
            if len(existing_pngs) >= 10:
                return (len(existing_pngs), len(existing_pngs))

    if not osp.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    png_output_dir = osp.join(target_dir, 'lip_mouth')
    os.makedirs(png_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0)

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()

    if frame_count == 0:
        return (0, 0)

    landmark_path = osp.join(landmarks_dir, 'landmarks.json')
    if not osp.exists(landmark_path):
        return (0, 0)

    with open(landmark_path, 'r') as f:
        landmarks_dict = json.load(f)

    landmarks = []
    for i in range(frame_count):
        frame_key = f'{i:04d}.jpg'
        if frame_key in landmarks_dict:
            lm = landmarks_dict[frame_key]
            if lm is not None:
                landmarks.append(np.array(lm))
            else:
                landmarks.append(None)
        else:
            landmarks.append(None)

    landmarks = landmarks_interpolate(landmarks)
    if landmarks is None:
        return (0, 0)

    mp4_output_path = osp.join(target_dir, 'speech_mouth.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(mp4_output_path, fourcc, 25.0, (args.crop_width, args.crop_height))

    q_frames, q_landmarks = deque(), deque()
    num_successful = 0
    output_frame_count = 0

    for i, frame in enumerate(frames):
        try:
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            q_frames.append(gray_frame)
            q_landmarks.append(landmarks[i])

            if len(q_frames) == args.window_margin:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)

                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frames.popleft()

                trans_frame, trans = warp_img(
                    smoothed_landmarks[STABLE_POINTS, :],
                    mean_face_landmarks[STABLE_POINTS, :],
                    cur_frame,
                    STD_SIZE
                )

                trans_landmarks = trans(cur_landmarks)

                cropped_frame = cut_patch(
                    trans_frame,
                    trans_landmarks[args.start_idx:args.stop_idx],
                    args.crop_height // 2,
                    args.crop_width // 2,
                )

                cropped_frame_bgr = cv2.cvtColor(cropped_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                video_writer.write(cropped_frame_bgr)

                cropped_frame_gray = Image.fromarray(cropped_frame.astype(np.uint8)).convert('L')
                png_path = osp.join(png_output_dir, f'{output_frame_count:04d}.png')
                cropped_frame_gray.save(png_path)

                output_frame_count += 1
                num_successful += 1

        except Exception as e:
            continue

    while q_frames:
        try:
            cur_frame = q_frames.popleft()
            cur_landmarks = q_landmarks.popleft()

            trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
            trans_landmarks = trans(cur_landmarks)

            cropped_frame = cut_patch(
                trans_frame,
                trans_landmarks[args.start_idx:args.stop_idx],
                args.crop_height // 2,
                args.crop_width // 2
            )

            cropped_frame_bgr = cv2.cvtColor(cropped_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            video_writer.write(cropped_frame_bgr)

            cropped_frame_gray = Image.fromarray(cropped_frame.astype(np.uint8)).convert('L')
            png_path = osp.join(png_output_dir, f'{output_frame_count:04d}.png')
            cropped_frame_gray.save(png_path)

            output_frame_count += 1
            num_successful += 1

        except Exception as e:
            continue

    video_writer.release()

    return (frame_count, num_successful)


def process_video_wrapper(args_tuple):
    video_rel_path, video_root, landmarks_root, output_root, mean_face_landmarks, args, skip_existing = args_tuple

    video_rel_path_no_ext = osp.splitext(video_rel_path)[0]

    video_path = osp.join(video_root, video_rel_path)
    landmarks_dir = osp.join(landmarks_root, video_rel_path_no_ext)
    target_dir = osp.join(output_root, video_rel_path_no_ext)

    if not osp.exists(video_path):
        return (video_rel_path, 0, 0)

    if not osp.exists(landmarks_dir):
        return (video_rel_path, 0, 0)

    num_frames, num_successful = crop_video_and_save(
        video_path, landmarks_dir, target_dir, mean_face_landmarks, args, skip_existing
    )

    return (video_rel_path, num_frames, num_successful)


def crop_mouths_main(args):
    print("="*60)
    print("Cropping Mouth Regions (Dual Output)")
    print("="*60)
    print(f"Video root: {args.video_root}")
    print(f"Landmarks directory: {args.landmarks_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Crop size: {args.crop_width}x{args.crop_height}")
    print(f"Window margin: {args.window_margin}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"\nOutput:")
    print(f"  - speech_mouth.mp4 (MP4 video)")
    print(f"  - lip_mouth/ (PNG sequence)")

    print("\nLoading mean face landmarks...")
    mean_face_landmarks = np.load(args.mean_face)
    print(f"Mean face shape: {mean_face_landmarks.shape}")

    video_list = []

    if args.file_list and osp.exists(args.file_list):
        print(f"\nLoading video list from: {args.file_list}")
        with open(args.file_list, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    if '\t' in line:
                        video_path = line.split('\t')[0]
                    else:
                        video_path = line.rsplit(maxsplit=1)[0] if ' ' in line else line.rsplit(',', 1)[0]
                    video_list.append(video_path)
    else:
        print(f"\nScanning landmarks directory for videos...")
        for root, dirs, files in os.walk(args.landmarks_dir):
            if 'landmarks.json' in files:
                rel_path = os.path.relpath(root, args.landmarks_dir)
                if rel_path == '.':
                    video_name = os.path.basename(args.landmarks_dir)
                else:
                    video_name = rel_path
                video_list.append(video_name + '.mp4')

    print(f"\nFound {len(video_list)} videos to process")
    if len(video_list) > 0:
        print(f"First few videos: {video_list[:3]}")

    process_args = [
        (video_rel_path, args.video_root, args.landmarks_dir, args.output_dir,
         mean_face_landmarks, args, args.skip_existing)
        for video_rel_path in video_list
    ]

    print("\nCropping mouth regions...")
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_video_wrapper, process_args),
            total=len(process_args),
            desc="Processing videos"
        ))

    total_frames = sum(num_frames for _, num_frames, _ in results)
    total_successful = sum(num_successful for _, _, num_successful in results)
    videos_processed = sum(1 for _, num_frames, _ in results if num_frames > 0)

    print(f"\n{'='*60}")
    print(f"Mouth Cropping Complete (Dual Output)")
    print(f"{'='*60}")
    print(f"Total videos: {len(results)}")
    print(f"Videos processed: {videos_processed}")
    print(f"Total frames: {total_frames}")
    print(f"Successfully cropped: {total_successful} ({100*total_successful/max(total_frames,1):.1f}%)")
    print(f"Failed: {total_frames - total_successful}")

    print(f"\nVideos with <80% success rate (first 20):")
    count = 0
    for video_path, num_frames, num_successful in results:
        if num_frames > 0 and num_successful / num_frames < 0.8:
            print(f"  - {video_path}: {num_successful}/{num_frames} ({100*num_successful/num_frames:.1f}%)")
            count += 1
            if count >= 20:
                remaining = sum(1 for v, nf, ns in results if nf > 0 and ns / nf < 0.8) - count
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break

    print(f"\n{'='*60}")
    print(f"Output Structure:")
    print(f"{'='*60}")
    print(f"  {args.output_dir}/video_name/")
    print(f"    speech_mouth.mp4")
    print(f"    lip_mouth/0000.png ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crop mouth regions from videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--video_root',
        type=str,
        required=True,
        help='Root directory of videos'
    )
    parser.add_argument(
        '--landmarks_dir',
        type=str,
        required=True,
        help='Directory containing landmarks'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for cropped mouths'
    )
    parser.add_argument(
        '--file_list',
        type=str,
        default=None,
        help='Path to file_list.txt (optional, for correct subdirectory handling)'
    )
    parser.add_argument(
        '--mean_face',
        type=str,
        default='data/20words_mean_face.npy',
        help='Path to mean face landmarks'
    )
    parser.add_argument(
        '--crop_width',
        type=int,
        default=96,
        help='Width of mouth ROI'
    )
    parser.add_argument(
        '--crop_height',
        type=int,
        default=96,
        help='Height of mouth ROI'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=48,
        help='Start index of mouth landmarks'
    )
    parser.add_argument(
        '--stop_idx',
        type=int,
        default=68,
        help='Stop index of mouth landmarks'
    )
    parser.add_argument(
        '--window_margin',
        type=int,
        default=12,
        help='Window size for landmark smoothing'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip videos that have already been processed'
    )

    args = parser.parse_args()
    crop_mouths_main(args)
