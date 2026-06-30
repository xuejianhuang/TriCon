import numpy as np
from skimage import transform as tf


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if int(center_y) - height < 0 - threshold:
        raise Exception("too much bias in height")

    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")

    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    img_cropped = np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )
    return img_cropped


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

    if not valid_frames_idx:
        return None

    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(
                landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
            )

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
            len(landmarks) - valid_frames_idx[-1]
        )

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"

    return landmarks
