import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import ndimage

from tqdm import tqdm

from data_loader_infer import (
    HaN_OAR_DICT,
    HaN_OAR_LIST,
    Lung_OAR_DICT,
    Lung_OAR_LIST,
)
from toolkit import combine_oar, combine_ptv


def _build_prescription_vectors(scale_entry: Dict[str, Dict]) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """Return prescribed dose vector plus dictionaries used for combining PTVs."""

    prescribed = [0.0, 0.0, 0.0]
    opt_dose_dict: Dict[str, float] = {}
    dose_dict: Dict[str, float] = {}

    order = ["PTV_High", "PTV_Mid", "PTV_Low"]
    for idx, key in enumerate(order):
        if key not in scale_entry:
            continue
        entry = scale_entry[key]
        pdose = float(entry["PDose"])
        prescribed[idx] = pdose

        opt_name = entry.get("OPTName")
        if opt_name:
            opt_dose_dict[opt_name] = pdose

        struct_name = entry.get("StructName")
        if key == "PTV_High":
            # High uses the same structure for both opt and actual dose in loader
            struct_key = opt_name
        else:
            struct_key = struct_name
        if struct_key:
            dose_dict[struct_key] = pdose

    return np.asarray(prescribed, dtype=np.float32), opt_dose_dict, dose_dict


def _ensure_angle_plate(data_dict: Dict[str, np.ndarray]) -> np.ndarray:
    angle_plate_3d = np.zeros_like(data_dict["img"], dtype=np.float32)
    isocenter = data_dict["isocenter"]

    z_begin = max(0, int(isocenter[0]) - 5)
    z_end = min(angle_plate_3d.shape[0], int(isocenter[0]) + 5)
    if z_end <= z_begin:
        z_begin = max(0, min(z_begin, angle_plate_3d.shape[0] - 1))
        z_end = min(angle_plate_3d.shape[0], z_begin + 1)

    if "angle_plate" not in data_dict:
        print("[WARN] angle_plate missing, using ones", flush=True)
        data_dict["angle_plate"] = np.ones(data_dict["img"][0].shape, dtype=np.float32)

    d3_plate = np.repeat(data_dict["angle_plate"][np.newaxis, :, :], max(1, z_end - z_begin), axis=0)
    if (
        d3_plate.shape[1] != angle_plate_3d.shape[1]
        or d3_plate.shape[2] != angle_plate_3d.shape[2]
    ):
        d3_plate = ndimage.zoom(
            d3_plate,
            (
                1,
                angle_plate_3d.shape[1] / d3_plate.shape[1],
                angle_plate_3d.shape[2] / d3_plate.shape[2],
            ),
            order=0,
        )
    angle_plate_3d[z_begin:z_end] = d3_plate
    return angle_plate_3d.astype(np.float32)


def _numpy_to_torch_channels(data_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    tensor_dict: Dict[str, torch.Tensor] = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            tensor_dict[key] = torch.from_numpy(value.astype(np.float32))[None]
    return tensor_dict


def preprocess_npz(
    npz_path: Path,
    output_path: Path,
    site: str,
    args,
    scale_dose_dict: Dict[str, Dict],
    pat_obj_dict: Dict[str, List[str]],
) -> None:
    with np.load(npz_path, allow_pickle=True) as npz_file:
        data_dict = dict(npz_file)["arr_0"].item()

    patient_token = npz_path.stem.split("+")[0]
    patient_id = patient_token if len(patient_token) >= 3 else f"{int(patient_token):03d}"

    # Basic normalization steps shared with the loaders
    data_dict["img"] = np.clip(
        data_dict["img"], args.down_hu, args.up_hu
    ).astype(np.float32) / float(args.denom_norm_hu)

    if "dose" in data_dict:
        scale_entry = scale_dose_dict.get(patient_id)
        if not scale_entry:
            print(f"[SKIP] Patient {patient_id} missing in scale_dose_dict for file {npz_path.name}")
            return
        ptv_highdose = float(scale_entry["PTV_High"]["PDose"])
        dose_scale = float(data_dict.get("dose_scale", 1.0))
        data_dict["dose"] = data_dict["dose"].astype(np.float32) * dose_scale
        ptv_mask_name = scale_entry["PTV_High"]["OPTName"]
        if ptv_mask_name not in data_dict:
            raise KeyError(f"{ptv_mask_name} missing in {npz_path}")
        mask = data_dict[ptv_mask_name].astype(bool)
        if not mask.any():
            raise ValueError(f"Mask {ptv_mask_name} empty in {npz_path}")
        d97 = np.percentile(data_dict["dose"][mask], 3) + 1e-5
        norm_scale = ptv_highdose / d97
        data_dict["dose"] = np.clip(
            data_dict["dose"] * norm_scale / float(args.dose_div_factor),
            0,
            ptv_highdose * 1.2,
        )

    data_dict["angle_plate"] = _ensure_angle_plate(data_dict)

    tensor_dict = _numpy_to_torch_channels(data_dict)

    # Determine site-specific metadata
    site_lower = site.lower()
    if site_lower == "han":
        oar_dict = HaN_OAR_DICT
        default_oars = HaN_OAR_LIST
    else:
        oar_dict = Lung_OAR_DICT
        default_oars = Lung_OAR_LIST

    oar_need_list = pat_obj_dict.get(patient_token, default_oars)
    oar_need_list = [roi for roi in oar_need_list if roi in oar_dict]

    if site_lower not in {"han", "lung"}:
        raise ValueError(f"Unsupported site: {site}")

    scale_entry = scale_dose_dict.get(patient_id)
    if not scale_entry:
        # already handled above; safe-guard to skip
        print(f"[SKIP] Patient {patient_id} missing in scale_dose_dict for file {npz_path.name}")
        return

    prescribed, opt_dose_dict, dose_dict = _build_prescription_vectors(scale_entry)
    comb_oar, _ = combine_oar(
        tensor_dict,
        oar_need_list,
        norm_oar=args.norm_oar,
        OAR_DICT=oar_dict,
    )
    comb_optptv, _, _ = combine_ptv(tensor_dict, opt_dose_dict)
    comb_ptv, _, _ = combine_ptv(tensor_dict, dose_dict)

    data_dict["comb_oar"] = comb_oar.squeeze(0).numpy().astype(np.float32)
    data_dict["comb_optptv"] = comb_optptv.squeeze(0).numpy().astype(np.float32)
    data_dict["comb_ptv"] = comb_ptv.squeeze(0).numpy().astype(np.float32)
    data_dict["prescribed_dose"] = prescribed
    data_dict["angle_plate"] = data_dict["angle_plate"].astype(np.float32)
    data_dict["beam_plate"] = data_dict["beam_plate"].astype(np.float32)
    data_dict["Body"] = data_dict["Body"].astype(np.float32)
    data_dict["img"] = data_dict["img"].astype(np.float32)
    if "dose" in data_dict:
        data_dict["dose"] = data_dict["dose"].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, arr_0=data_dict)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute combined channels for GDP datasets")
    parser.add_argument("--datasets-root", default="datasets", help="Root directory containing site folders")
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["HaN", "Lung"],
        help="Sites to preprocess",
    )
    parser.add_argument("--split", default="train_raw", help="Dataset split to read (e.g., train)")
    parser.add_argument(
        "--output-suffix",
        default="train",
        help="Name of the output folder for processed npz files",
    )
    parser.add_argument("--scale-dose-dict", default="meta_files/PTV_DICT.json")
    parser.add_argument("--pat-obj-dict", default="meta_files/Pat_Obj_DICT.json")
    parser.add_argument("--down-hu", type=int, default=-1000)
    parser.add_argument("--up-hu", type=int, default=1000)
    parser.add_argument("--denom-norm-hu", type=float, default=500.0)
    parser.add_argument("--dose-div-factor", type=float, default=10.0)
    parser.add_argument(
        "--no-norm-oar",
        action="store_false",
        dest="norm_oar",
        help="Disable OAR normalization used during combination",
    )
    parser.set_defaults(norm_oar=True)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in the output folder",
    )
    args = parser.parse_args()

    with open(args.scale_dose_dict, "r", encoding="utf-8") as f:
        scale_dose_dict = json.load(f)
    with open(args.pat_obj_dict, "r", encoding="utf-8") as f:
        pat_obj_dict = json.load(f)

    for site in args.sites:
        input_dir = Path(args.datasets_root) / site / args.split
        output_dir = Path(args.datasets_root) / site / args.output_suffix

        if not input_dir.exists():
            print(f"[WARN] Input directory {input_dir} not found, skipping")
            continue

        npz_files = sorted(input_dir.glob("*.npz"))
        print(f"Processing {len(npz_files)} files for site {site}")
        for npz_path in tqdm(npz_files, desc=f"Site {site}"):
            output_path = output_dir / npz_path.name
            if args.skip_existing and output_path.exists():
                continue
            preprocess_npz(npz_path, output_path, site, args, scale_dose_dict, pat_obj_dict)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
