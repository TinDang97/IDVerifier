import torch

from util.face_align_trans import warp_and_crop_face
from .RetinaFace import RetinaFace
from model.retinaface import cfg_re50
from torchvision import models


class FaceDetector:
    def __init__(self, model_path, threshold=0.5, face_size=(112, 112)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.__detector = RetinaFace(models.resnet50, cfg_re50, self.device)

        print('Face Detector: Loading detector from {}'.format(model_path))
        self.__detector.load_model(model_path)

        from util.face_align_trans import get_reference_facial_points
        self.__reference = get_reference_facial_points(default_square=True)
        self.threshold = threshold
        self.face_size = face_size

    def detect(self, image):
        result = []
        for face_detail in self.__detector.detect(image, self.threshold):
            border = [idx for idx, pos in enumerate(face_detail[:4]) if pos > 5000 or pos < 0]

            if len(border) > 0:
                continue
            box = tuple(face_detail[:4])
            land_mask = face_detail[5:].reshape((5, 2))
            result.append((land_mask, box))
        torch.cuda.empty_cache()
        return result

    def warp_crop_face(self, image, land_mask, output_size=None):
        if output_size is None:
            output_size = self.face_size
        return warp_and_crop_face(image, land_mask, self.__reference, output_size)