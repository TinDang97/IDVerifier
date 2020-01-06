import flask
import argparse
import os
import cv2
import numpy
import torch
from flask import request

from FaceDetector.FaceDetector import FaceDetector
from FaceEmbedding.FaceEmbedder import FaceEmbedder
from util.util import cosine_similarity, resize_image

argparser = argparse.ArgumentParser(description="Toolkit uses to verify 2 person is one.")
argparser.add_argument('--detect-model-path', type=str, default="./data/retina_res50.pth", help="Path of image compare")
argparser.add_argument('--detect-threshold', type=float, default=0.8, help="Path of image compare")
argparser.add_argument('--port', type=int, default=1280, help="Port use to binding.")
argparser.add_argument('--embed-model-path', type=str, default="./data/backbone_ir50_asia.pth", help="Path of image compare")
args = argparser.parse_args()

if __name__ == '__main__':
    assert os.path.isfile(args.detect_model_path) and os.path.isfile(args.embed_model_path), "Model's file not found!"

    face_detector = FaceDetector(args.detect_model_path, threshold=args.detect_threshold)
    face_embedding = FaceEmbedder(args.embed_model_path)

    app = flask.Flask(__name__)

    def decode_image(binary):
        binary = numpy.fromstring(binary, numpy.uint8)
        return cv2.imdecode(binary, cv2.IMREAD_ANYCOLOR)

    @app.route("/verify_person/", methods=['POST'])
    def verify():
        try:
            try:
                image_base = resize_image(decode_image(request.files['image_base'].read()), (1280, 720))
                image_compare = resize_image(decode_image(request.files['image_compare'].read()), (1280, 720))
            except Exception:
                return "Method require 2 params: image_base, image_compare. Please re-check it."
            face_base = face_detector.detect(image_base)
            face_compare = face_detector.detect(image_compare)

            assert len(face_base) == 1 and len(face_compare) == 1, "Only one face in image!"

            faces = [face_detector.warp_crop_face(image_base, face_base[0][0]),
                     face_detector.warp_crop_face(image_compare, face_compare[0][0])]

            embs = face_embedding.get_features(faces)

            cos_sim = cosine_similarity(embs[0], embs[1])
            output = f"score: {cos_sim * 100}%"

            print(output)
            return output
        except Exception:
            return "Server have error!"
        finally:
            torch.cuda.empty_cache()

    app.run("0.0.0.0", args.port)
