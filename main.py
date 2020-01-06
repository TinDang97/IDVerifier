import argparse
import os
import cv2

from FaceDetector.FaceDetector import FaceDetector
from FaceEmbedding.FaceEmbedder import FaceEmbedder
from util.util import cosine_similarity, resize_image

argparser = argparse.ArgumentParser(description="Toolkit uses to verify 2 person is one.")
argparser.add_argument('--image-base', type=str, required=True, help="Path of image base")
argparser.add_argument('--image-compare', type=str, required=True, help="Path of image compare")
argparser.add_argument('--detect-model-path', type=str, default="./data/retina_res50.pth", help="Path of image compare")
argparser.add_argument('--detect-threshold', type=float, default=0.8, help="Path of image compare")
argparser.add_argument('--embed-model-path', type=str, default="./data/backbone_ir50_asia.pth", help="Path of image compare")
args = argparser.parse_args()

if __name__ == '__main__':
    assert os.path.isfile(args.image_base) and os.path.isfile(args.image_compare), "File not found!"
    assert os.path.isfile(args.detect_model_path) and os.path.isfile(args.embed_model_path), "Model's file not found!"

    face_detector = FaceDetector(args.detect_model_path, threshold=args.detect_threshold)
    face_embedding = FaceEmbedder(args.embed_model_path)

    image_base = resize_image(cv2.imread(args.image_base), (1280, 720))
    image_compare = resize_image(cv2.imread(args.image_compare), (1280, 720))

    face_base = face_detector.detect(image_base)
    face_compare = face_detector.detect(image_compare)

    assert len(face_base) == 1 and len(face_compare) == 1, "Only one face in image!"

    faces = [face_detector.warp_crop_face(image_base, face_base[0][0]),
             face_detector.warp_crop_face(image_compare, face_compare[0][0])]

    embs = face_embedding.get_features(faces)

    cos_sim = cosine_similarity(embs[0], embs[1])

    print(f"score: {cos_sim * 100}")
