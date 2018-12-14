import torch

from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
}


def predict_collate_fn(batch):
    images = []
    boxes = []
    labels = []
    for image, box, label in batch:
        images.append(image)
        boxes.append(box)
        labels.append(label)
    images = torch.stack(images, dim=0)
    return images, boxes, labels


def build_dataset(dataset_list, transform=None, target_transform=None, is_test=False):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        if is_test and factory == VOCDataset:
            args['keep_difficult'] = True
        args['transform'] = transform
        args['target_transform'] = target_transform
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if is_test:
        return datasets
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    return dataset
