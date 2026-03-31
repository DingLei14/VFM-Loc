from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import build_eval_transforms


def _read_rgb(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class CVUSAEvalDataset(Dataset):
    def __init__(self, data_root, split, img_type, transforms=None):
        self.data_root = data_root
        self.transforms = transforms
        csv_name = "train-19zl.csv" if split == "train" else "val-19zl.csv"
        df = pd.read_csv(f"{data_root}/splits/{csv_name}", header=None)
        df = df.rename(columns={0: "sat", 1: "ground"})
        df["idx"] = df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        if img_type == "reference":
            self.images = df.sat.tolist()
            self.labels = df.idx.astype(int).tolist()
        else:
            self.images = df.ground.tolist()
            self.labels = df.idx.astype(int).tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = _read_rgb(f"{self.data_root}/{self.images[index]}")
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, torch.tensor(self.labels[index], dtype=torch.long)


class VigorEvalDataset(Dataset):
    def __init__(self, data_root, split, img_type, same_area=True, transforms=None):
        self.transforms = transforms
        cities = ["Chicago", "NewYork", "SanFrancisco", "Seattle"] if same_area else (["NewYork", "Seattle"] if split == "train" else ["Chicago", "SanFrancisco"])

        sat_frames = []
        for city in cities:
            df_city = pd.read_csv(f"{data_root}/splits/{city}/satellite_list.txt", header=None, sep=r"\s+")
            df_city = df_city.rename(columns={0: "sat"})
            df_city["path"] = df_city.sat.map(lambda name: f"{data_root}/satellite/{city}/{name}")
            sat_frames.append(df_city)
        df_sat = pd.concat(sat_frames, axis=0).reset_index(drop=True)
        sat2idx = dict(zip(df_sat.sat, df_sat.index))
        idx2sat_path = dict(zip(df_sat.index, df_sat.path))

        ground_frames = []
        split_name = f"same_area_balanced_{split}.txt" if same_area else "pano_label_balanced.txt"
        for city in cities:
            df_city = pd.read_csv(f"{data_root}/splits/{city}/{split_name}", header=None, sep=r"\s+")
            df_city = df_city.loc[:, [0, 1, 4, 7, 10]].rename(columns={0: "ground", 1: "sat", 4: "sat_np1", 7: "sat_np2", 10: "sat_np3"})
            df_city["path_ground"] = df_city.ground.map(lambda name: f"{data_root}/ground/{city}/{name}")
            for key in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_city[key] = df_city[key].map(sat2idx)
            ground_frames.append(df_city)
        df_ground = pd.concat(ground_frames, axis=0).reset_index(drop=True)

        if img_type == "reference":
            if split == "train":
                labels = df_ground["sat"].unique().tolist()
                self.images = [idx2sat_path[idx] for idx in labels]
                self.labels = labels
            else:
                self.images = df_sat.path.tolist()
                self.labels = df_sat.index.astype(int).tolist()
        else:
            self.images = df_ground.path_ground.tolist()
            self.labels = df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = _read_rgb(self.images[index])
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, torch.tensor(self.labels[index], dtype=torch.long)


class CVACTEvalDataset(Dataset):
    def __init__(self, data_root, split, img_type, transforms=None):
        self.transforms = transforms
        self.img_type = img_type
        self.data_root = data_root

        anu_data = sio.loadmat(f"{data_root}/ACT_data.mat")
        ids = anu_data["panoIds"][anu_data[f"{split}Set"][0][0][1] - 1]
        self.samples = []
        self.idx2label = {}
        counter = 0
        for idx in ids.squeeze():
            idx = str(idx)
            grd = f"{data_root}/ANU_data_small/streetview/{idx}_grdView.jpg"
            sat = f"{data_root}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg"
            if os.path.exists(grd) and os.path.exists(sat):
                self.idx2label[idx] = counter
                self.samples.append(idx)
                counter += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        idx = self.samples[index]
        if self.img_type == "reference":
            path = f"{self.data_root}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg"
        else:
            path = f"{self.data_root}/ANU_data_small/streetview/{idx}_grdView.jpg"
        img = _read_rgb(path)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, torch.tensor(self.idx2label[idx], dtype=torch.long)


def _list_id_dirs(folder: str) -> list[str]:
    return sorted([name for name in os.listdir(folder) if (Path(folder) / name).is_dir()])


def _collect_university_data(folder: str) -> Dict[str, dict]:
    data = {}
    for folder_name in _list_id_dirs(folder):
        dir_path = Path(folder) / folder_name
        files = sorted([file for file in os.listdir(dir_path) if (dir_path / file).is_file()])
        data[folder_name] = {"path": str(dir_path), "files": files}
    return data


class UniversityEvalDataset(Dataset):
    def __init__(self, data_folder, transforms=None, sample_ids: Optional[Set[str]] = None):
        self.transforms = transforms
        self.data_dict = _collect_university_data(data_folder)
        self.images = []
        self.labels = []
        self.sample_ids_raw = []
        for sample_id, item in self.data_dict.items():
            for file_name in item["files"]:
                self.images.append(f"{item['path']}/{file_name}")
                self.sample_ids_raw.append(sample_id)
                self.labels.append(-1 if sample_ids is not None and sample_id not in sample_ids else int(sample_id))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = _read_rgb(self.images[index])
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, torch.tensor(self.labels[index], dtype=torch.long)

    def get_sample_ids(self) -> Set[str]:
        return set(self.sample_ids_raw)


def get_model_transforms(config: Dict, preprocess: Dict):
    dataset_cfg = config["dataset"]
    reference_size = tuple(dataset_cfg["reference_size"])
    query_size = tuple(dataset_cfg["query_size"])
    ground_cutting = int(dataset_cfg.get("ground_cutting", 0))
    eval_reference, eval_query = build_eval_transforms(
        reference_size,
        query_size,
        preprocess["mean"],
        preprocess["std"],
        ground_cutting=ground_cutting,
    )
    return {
        "eval_reference": eval_reference,
        "eval_query": eval_query,
    }


def build_eval_datasets(config: Dict, transforms: Dict):
    dataset_cfg = config["dataset"]
    name = dataset_cfg["name"].lower()
    data_root = dataset_cfg["data_root"]
    split = dataset_cfg.get("eval_split", "test")

    if name == "cvusa":
        reference_ds = CVUSAEvalDataset(data_root, split=split, img_type="reference", transforms=transforms["eval_reference"])
        query_ds = CVUSAEvalDataset(data_root, split=split, img_type="query", transforms=transforms["eval_query"])
        return query_ds, reference_ds

    if name == "vigor":
        same_area = bool(dataset_cfg.get("same_area", True))
        reference_ds = VigorEvalDataset(data_root, split=split, img_type="reference", same_area=same_area, transforms=transforms["eval_reference"])
        query_ds = VigorEvalDataset(data_root, split=split, img_type="query", same_area=same_area, transforms=transforms["eval_query"])
        return query_ds, reference_ds

    if name == "cvact":
        reference_ds = CVACTEvalDataset(data_root, split=split, img_type="reference", transforms=transforms["eval_reference"])
        query_ds = CVACTEvalDataset(data_root, split=split, img_type="query", transforms=transforms["eval_query"])
        return query_ds, reference_ds

    if name == "u1652":
        task = dataset_cfg.get("task", "D2S").upper()
        if task == "D2S":
            query_folder = f"{data_root}/{split}/query_drone"
            gallery_folder = f"{data_root}/{split}/gallery_satellite"
        else:
            query_folder = f"{data_root}/{split}/query_satellite"
            gallery_folder = f"{data_root}/{split}/gallery_drone"
        query_ds_temp = UniversityEvalDataset(query_folder, transforms=None)
        gallery_ds_temp = UniversityEvalDataset(gallery_folder, transforms=None)
        common_ids = query_ds_temp.get_sample_ids().intersection(gallery_ds_temp.get_sample_ids())
        if not common_ids:
            raise RuntimeError("No common IDs between University query and gallery folders.")
        query_ds = UniversityEvalDataset(query_folder, transforms=transforms["eval_query"], sample_ids=common_ids)
        gallery_ds = UniversityEvalDataset(gallery_folder, transforms=transforms["eval_reference"], sample_ids=common_ids)
        return query_ds, gallery_ds

    raise ValueError(f"Unsupported dataset: {name}")


def build_eval_loaders(config: Dict, query_ds: Dataset, reference_ds: Dataset) -> Tuple[DataLoader, DataLoader]:
    evaluation = config["evaluation"]
    query_loader = DataLoader(
        query_ds,
        batch_size=int(evaluation.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(evaluation.get("num_workers", 4)),
        pin_memory=True,
    )
    reference_loader = DataLoader(
        reference_ds,
        batch_size=int(evaluation.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(evaluation.get("num_workers", 4)),
        pin_memory=True,
    )
    return query_loader, reference_loader
