from __future__ import annotations

from typing import Iterable, Tuple

import albumentations as A
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class Cut(ImageOnlyTransform):
    def __init__(self, cutting: int = 0, p: float = 1.0):
        super().__init__(p=p)
        self.cutting = int(cutting)

    def apply(self, image, **params):
        if self.cutting > 0:
            return image[self.cutting : -self.cutting, :, :]
        return image

    def get_transform_init_args_names(self):
        return ("cutting",)


def _normalize(mean: Iterable[float], std: Iterable[float]) -> list:
    return [A.Normalize(mean=mean, std=std), ToTensorV2()]


def build_eval_transforms(
    reference_size: Tuple[int, int],
    query_size: Tuple[int, int],
    mean,
    std,
    ground_cutting: int = 0,
):
    reference = A.Compose(
        [A.Resize(reference_size[0], reference_size[1], interpolation=cv2.INTER_LINEAR_EXACT)] + _normalize(mean, std)
    )
    query = A.Compose(
        [Cut(cutting=ground_cutting), A.Resize(query_size[0], query_size[1], interpolation=cv2.INTER_LINEAR_EXACT)]
        + _normalize(mean, std)
    )
    return reference, query
