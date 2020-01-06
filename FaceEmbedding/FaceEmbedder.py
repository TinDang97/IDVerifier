import cv2
import numpy
import torch
from torchvision import transforms
from .InsightFace import InsightFace
from model.model_irse import IR_50


class FaceEmbedder(object):
    def __init__(self, model_path, input_size=(112, 112), backbone=IR_50, model=InsightFace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load embedding model
        self.embedder = model(backbone(input_size))
        self.embedder.load_model(model_path, self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],  # RGB_MEAN
                [0.5, 0.5, 0.5]),  # RGB_STD
        ])

    def get_features(self, faces):
        """
        Get embedding features of faces
        :param faces: face list
        :return: feature list
        """
        transform = self.transform
        device = self.device
        faces_tranformed = []
        faces_mirror_tranformed = []

        for face in faces:
            face_mirror = cv2.cvtColor(cv2.flip(face.copy(), 1), cv2.COLOR_BGR2RGB)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face_mirror = transform(face_mirror).detach().numpy()
            face = transform(face).detach().numpy()

            faces_tranformed.append(face)
            faces_mirror_tranformed.append(face_mirror)

        embs = self.embedder.get_feature(
            torch.from_numpy(numpy.array(faces_tranformed + faces_mirror_tranformed)).to(device)
        )
        result = (embs[:len(faces)] + embs[len(faces):]).cpu().detach().numpy().astype(numpy.float32)
        torch.cuda.empty_cache()
        return result
