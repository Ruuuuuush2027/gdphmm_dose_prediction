
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import json

from scipy import ndimage
from os.path import join
from monai.transforms import (
    Compose,
    Resized,
    RandFlipd,
    RandRotated,
    SpatialPadd,
    SpatialCropd,
    RandSpatialCropd,
)

def tr_augmentation_v1(KEYS, in_size, out_size, crop_center):

    return Compose(
        [
            SpatialCropd(
                keys=KEYS,
                roi_center=crop_center,
                roi_size=[
                    int(in_size[0] * 1.2),
                    int(in_size[1] * 1.2),
                    int(in_size[2] * 1.2),
                ],
                allow_missing_keys=True,
            ),
            SpatialPadd(
                keys=KEYS,
                spatial_size=[
                    int(in_size[0] * 1.2),
                    int(in_size[1] * 1.2),
                    int(in_size[2] * 1.2),
                ],
                mode="constant",
                allow_missing_keys=True,
            ),
            RandSpatialCropd(
                keys=KEYS,
                roi_size=[
                    int(in_size[0] * 0.85),
                    int(in_size[1] * 0.85),
                    int(in_size[2] * 0.85),
                ],
                max_roi_size=[
                    int(in_size[0] * 1.2),
                    int(in_size[1] * 1.2),
                    int(in_size[2] * 1.2),
                ],
                random_center=True,
                random_size=True,
                allow_missing_keys=True,
            ),
            # AUG_KEYS =['comb_optptv', 'comb_ptv', 'comb_oar', 'Body',
            # 'img', 'dose',  'beam_plate', 'angle_plate']
            RandRotated(
                keys=KEYS,
                prob=0.8,
                range_x=1,
                range_y=0.2,
                range_z=0.2,
                allow_missing_keys=True,
                mode=(
                    "nearest",
                    "nearest",
                    "nearest",
                    "nearest",
                    "bilinear",
                    "bilinear",
                    "bilinear",
                    "nearest",
                ),
            ),
            RandFlipd(keys=KEYS, prob=0.4, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=KEYS, prob=0.4, spatial_axis=1, allow_missing_keys=True),
            RandFlipd(keys=KEYS, prob=0.4, spatial_axis=2, allow_missing_keys=True),
            Resized(
                keys=KEYS,
                spatial_size=out_size,
                allow_missing_keys=True,
                mode=(
                    "nearest",
                    "nearest",
                    "nearest",
                    "nearest",
                    "area",
                    "area",
                    "area",
                    "nearest",
                ),
            ),
        ]
    )


def tt_augmentation_v1(KEYS, in_size, out_size, crop_center):
    return Compose(
        [
            SpatialCropd(
                keys=KEYS,
                roi_center=crop_center,
                roi_size=in_size,
                allow_missing_keys=True,
            ),
            SpatialPadd(
                keys=KEYS,
                spatial_size=in_size,
                mode="constant",
                allow_missing_keys=True,
            ),
            Resized(keys=KEYS, spatial_size=out_size, allow_missing_keys=True,
                                    mode=(
                    "nearest",
                    "nearest",
                    "nearest",
                    "nearest",
                    "area",
                    "area",
                    "area",
                    "nearest",
                ),)
        ]
    )


class GDPDataset_v2(Dataset):

    def __init__(self, cfig, phase, site, debug=False):
        """
        phase: train, valid, or test
        cfig: the configuration dictionary
        """

        self.cfig = cfig

        df = pd.read_csv(cfig["csv_root"])
        
        if phase == "val":
            phase = "valid"
        df = df.loc[df["dev_split"] == phase]
        if site == "HaN":
            site_index = 1
        elif site == "Lung":
            site_index = 2
        
        
        df = df.loc[df["site"] == site_index]
        
        self.phase = phase
        self.data_list = df["npz_path"].tolist()

        self.site_list = df["site"].tolist()
        self.cohort_list = df["cohort"].tolist()

        self.scale_dose_Dict = json.load(open(cfig["scale_dose_dict"], "r"))
        self.pat_obj_dict = json.load(open(cfig["pat_obj_dict"], "r"))
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_path = self.data_list[index]
        ID = self.data_list[index].split("/")[-1].replace(".npz", "")
        PatientID = ID.split("+")[0]

        if len(str(PatientID)) < 3:
            PatientID = f"{PatientID:0>3}"

        current_data_path = join(
            os.environ["Dataset"],
            data_path,
        )

        # *############################# img and dose  #############################
        data_npz = np.load(current_data_path, allow_pickle=True)

        In_dict = dict(data_npz)["arr_0"].item()
        In_dict["img"] = (
            np.clip(In_dict["img"], self.cfig["down_HU"], self.cfig["up_HU"])
            / self.cfig["denom_norm_HU"]
        )

        ori_img_size = In_dict["img"].shape

        if "dose" in In_dict.keys():

            In_dict["dose"] = In_dict["dose"] * In_dict["dose_scale"]

            ptv_highdose = self.scale_dose_Dict[PatientID]["PTV_High"]["PDose"]
            PTVHighOPT = self.scale_dose_Dict[PatientID]["PTV_High"]["OPTName"]

            norm_scale = ptv_highdose / (
                np.percentile(In_dict["dose"][In_dict[PTVHighOPT].astype("bool")], 3)
                + 1e-5
            )  # D97

            In_dict["dose"] = (
                In_dict["dose"] * norm_scale / self.cfig["dose_div_factor"]
            )
            In_dict["dose"] = np.clip(In_dict["dose"], 0, ptv_highdose * 1.2)

        isocenter = In_dict["isocenter"]

        # *############################# angle plate  #############################
        angle_plate_3D = np.zeros(In_dict["img"].shape)

        z_begin = int(isocenter[0]) - 5
        z_end = int(isocenter[0]) + 5
        z_begin = max(0, z_begin)
        z_end = min(angle_plate_3D.shape[0], z_end)

        if "angle_plate" not in In_dict.keys():
            print(" **************** angle_plate error in ", data_path)
            In_dict["angle_plate"] = np.ones(In_dict["img"][0].shape)

        if In_dict["angle_plate"].ndim == 2:
            D3_plate = np.repeat(
                In_dict["angle_plate"][np.newaxis, :, :], max(1, z_end - z_begin), axis=0
            )
        else:
            D3_plate = In_dict["angle_plate"][z_begin:z_end]

        if (
            D3_plate.shape[1] != angle_plate_3D.shape[1]
            or D3_plate.shape[2] != angle_plate_3D.shape[2]
        ):
            # breakpoint()
            D3_plate = ndimage.zoom(
                D3_plate,
                (
                    1,
                    angle_plate_3D.shape[1] / D3_plate.shape[1],
                    angle_plate_3D.shape[2] / D3_plate.shape[2],
                ),
                order=0,
            )
        angle_plate_3D[z_begin:z_end] = D3_plate

        In_dict["angle_plate"] = angle_plate_3D
        In_dict["comb_optptv"] = In_dict["comb_optptv"] / self.cfig["dose_div_factor"]
        In_dict["comb_ptv"] = In_dict["comb_ptv"] / self.cfig["dose_div_factor"]

        # *############################# augmentation  #############################
        AUG_KEYS = [
            "comb_optptv",
            "comb_ptv",
            "comb_oar",
            "Body",
            "img",
            "dose",
            "beam_plate",
            "angle_plate",
        ]

        for key in AUG_KEYS:
            assert (
                isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3
            ), f"key {key} not in In_dict or not np.ndarray or not float32"

            In_dict[key] = torch.from_numpy(In_dict[key].astype("float"))[None]

        if self.phase == "train":
            self.aug = tr_augmentation_v1(
                AUG_KEYS, self.cfig["in_size"], self.cfig["out_size"], isocenter
            )
        else:
            self.aug = tt_augmentation_v1(
                AUG_KEYS, self.cfig["in_size"], self.cfig["out_size"], isocenter
            )
        In_dict = self.aug(In_dict)

        # *############################# debug 2 #############################
      

        # *############################# final output #############################
        data_dict = dict()

        if "dose" in In_dict.keys():
            data_dict["label"] = In_dict["dose"] * In_dict["Body"]

        data_dict["data"] = torch.cat(
            (
                In_dict["comb_optptv"],
                In_dict["comb_ptv"],
                In_dict["comb_oar"],
                In_dict["Body"],
                In_dict["img"],
                In_dict["beam_plate"],
                In_dict["angle_plate"],
            ),
            axis=0,
        )  #

        data_dict["prescribed_dose"] = In_dict["prescribed_dose"]
        data_dict["id"] = ID
        data_dict["ori_img_size"] = torch.tensor(ori_img_size)
        data_dict["ori_isocenter"] = torch.tensor(isocenter)

        del In_dict

        return data_dict


if __name__ == "__main__":
    # train 2725 val 150 test: 356
    cfig = {
        "train_bs": 1,
        "val_bs": 1,
        "num_workers": 1,
        "csv_root": "meta_files/meta_data.csv",
        "scale_dose_dict": "meta_files/PTV_DICT.json",
        "pat_obj_dict": "meta_files/Pat_Obj_DICT.json",
        "down_HU": -1000,
        "up_HU": 1000,
        "denom_norm_HU": 500,
        "in_size": (96, 128, 144),
        "out_size": (96, 128, 144),
        "norm_oar": True,
        "CatStructures": False,
        "dose_div_factor": 10,
    }

    if 0:
        train_dataset = GDPDataset_v2(cfig, phase="train", site="Lung", debug=False)
        val_dataset = GDPDataset_v2(cfig, phase="val", site="Lung", debug=False)
        print("len(train_dataset): ", len(train_dataset))
        print("len(val_dataset): ", len(val_dataset))        

    elif 1:
        
        train_HaN_dataset = GDPDataset_v2(cfig, phase="train", site="HaN", debug=False)
        val_HaN_dataset = GDPDataset_v2(cfig, phase="val", site="HaN", debug=False)
        train_Lung_dataset = GDPDataset_v2(cfig, phase="train", site="Lung", debug=False)
        val_Lung_dataset = GDPDataset_v2(cfig, phase="val", site="Lung", debug=False)
        
        print("len(train_HaN_dataset): ", len(train_HaN_dataset))
        print("len(val_HaN_dataset): ", len(val_HaN_dataset))
        print("len(train_Lung_dataset): ", len(train_Lung_dataset))
        print("len(val_Lung_dataset): ", len(val_Lung_dataset))
        
    
  