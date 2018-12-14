import torch

from ssd.modeling.post_processor import PostProcessor


class Predictor:
    def __init__(self, cfg, model, iou_threshold, score_threshold, device):
        self.cfg = cfg
        self.model = model
        self.post_processor = PostProcessor(iou_threshold=iou_threshold,
                                            score_threshold=score_threshold,
                                            image_size=cfg.INPUT.IMAGE_SIZE,
                                            max_per_class=cfg.TEST.MAX_PER_CLASS,
                                            max_per_image=cfg.TEST.MAX_PER_IMAGE)
        self.device = device
        self.model.eval()

    def predict(self, images):
        """predict batched images
        Args:
            images: [n, 3, size, size]
        Returns:
            List[(boxes, labels, scores)],
            boxes: (n, 4)
            labels: (n, )
            scores: (n, )
        """
        images = images.to(self.device)
        with torch.no_grad():
            scores, boxes = self.model(images)
        results = self.post_processor(scores, boxes)
        return results
