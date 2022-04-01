import bz2
import os
import os.path as osp
import sys
from multiprocessing import Pool

import dlib
import numpy as np
import PIL.Image
import requests
import scipy.ndimage
from tqdm import tqdm
from argparse import ArgumentParser

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def image_align(src_file,
                dst_file,
                face_landmarks,
                output_size=1024,
                transform_size=4096,
                enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print(
            '\nCannot find source image. Please run "--wilds" before "--align".'
        )
        return
    img = PIL.Image.open(src_file)
    img = img.convert('RGB')

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                  'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'PNG')


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector(
        )  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [
                (item.x, item.y)
                for item in self.shape_predictor(img, detection).parts()
            ]
            yield face_landmarks


def unpack_bz2(src_path):
    dst_path = src_path[:-4]
    if os.path.exists(dst_path):
        print('cached')
        return dst_path
    data = bz2.BZ2File(src_path).read()
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def work_landmark(raw_img_path, img_name, face_landmarks):
    face_img_name = '%s.png' % (os.path.splitext(img_name)[0], )
    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
    if os.path.exists(aligned_face_path):
        return
    image_align(raw_img_path,
                aligned_face_path,
                face_landmarks,
                output_size=256)


def get_file(src, tgt):
    if os.path.exists(tgt):
        print('cached')
        return tgt
    tgt_dir = os.path.dirname(tgt)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    file = requests.get(src)
    open(tgt, 'wb').write(file.content)
    return tgt


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input_imgs_path",
                        type=str,
                        default="imgs",
                        help="input images directory path")
    parser.add_argument("-o",
                        "--output_imgs_path",
                        type=str,
                        default="imgs_align",
                        help="output images directory path")

    args = parser.parse_args()

    # takes very long time  ...
    landmarks_model_path = unpack_bz2(
        get_file(
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'temp/shape_predictor_68_face_landmarks.dat.bz2'))

    # RAW_IMAGES_DIR = sys.argv[1]
    # ALIGNED_IMAGES_DIR = sys.argv[2]
    RAW_IMAGES_DIR = args.input_imgs_path
    ALIGNED_IMAGES_DIR = args.output_imgs_path

    if not osp.exists(ALIGNED_IMAGES_DIR): os.makedirs(ALIGNED_IMAGES_DIR)

    files = os.listdir(RAW_IMAGES_DIR)
    print(f'total img files {len(files)}')
    with tqdm(total=len(files)) as progress:

        def cb(*args):
            # print('update')
            progress.update()

        def err_cb(e):
            print('error:', e)

        with Pool(8) as pool:
            res = []
            landmarks_detector = LandmarksDetector(landmarks_model_path)
            for img_name in files:
                raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
                # print('img_name:', img_name)
                for i, face_landmarks in enumerate(
                        landmarks_detector.get_landmarks(raw_img_path),
                        start=1):
                    # assert i == 1, f'{i}'
                    # print(i, face_landmarks)
                    # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    # aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    # image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=256)

                    work_landmark(raw_img_path, img_name, face_landmarks)
                    progress.update()

                    # job = pool.apply_async(
                    #     work_landmark,
                    #     (raw_img_path, img_name, face_landmarks),
                    #     callback=cb,
                    #     error_callback=err_cb,
                    # )
                    # res.append(job)

            # pool.close()
            # pool.join()
    print(f"output aligned images at: {ALIGNED_IMAGES_DIR}")
