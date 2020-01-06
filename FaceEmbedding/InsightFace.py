import torch


class InsightFace(object):
    def __init__(self, backbone):
        self.backbone = backbone

    def load_model(self, pretrained_path, device):
        assert pretrained_path, "Pre-trained model is not found!"

        self.backbone.to(device)
        self.backbone.load_state_dict(torch.load(pretrained_path))
        self.backbone.eval()

    def get_feature(self, face):
        return self.backbone(face)
