import numpy as np
import os
from os.path import join

def dose_MAE_metric(pred_dose, pred_id):
   
    current_data_folder = join(
        os.environ["Dataset"],
        "GDP",
        "val_v1")
    
    
    ref_dose = np.load(join(current_data_folder, pred_id+"_dose.npy"))
    ref_Body = np.load(join(current_data_folder, pred_id+"_Body.npy"))
    assert ref_dose.shape == ref_Body.shape == pred_dose.shape, "Shape mismatch"
    
    # Calculate the Mean Absolute Error (MAE)
    
    # The mask include the body AND the region where the dose/prediction is higher than 5Gy
    isodose_5Gy_mask = ((ref_dose > 5) | (pred_dose > 5)) & (ref_Body > 0) 
    
    # The mask include the body AND the region where the ref is higher than 5Gy
    isodose_ref_5Gy_mask = (ref_dose > 5) & (ref_Body > 0) 

    diff = ref_dose - pred_dose

    error = np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)
    
    return error



def offset_spatial_crop(roi_center=None, roi_size=None):
    """
    for crop spatial regions of the data based on the specified `roi_center` and `roi_size`.

    get the start and end of the crop

    Parameters:
        roi_center (tuple of int, optional): The center point of the region of interest (ROI).
        roi_size (tuple of int, optional): The size of the ROI in each spatial dimension.

    Returns:
        start & end: start and end offsets
    """

    if roi_center is None or roi_size is None:
        raise ValueError("Both `roi_center` and `roi_size` must be specified.")

    roi_center = [int(round(c)) for c in roi_center]
    roi_size = [int(round(s)) for s in roi_size]

    start = []
    end = []

    for i, (center, size) in enumerate(zip(roi_center, roi_size)):

        half_size = size // 2  # int(round(size / 2))
        start_i = max(center - half_size, 0)  # Ensure we don't go below 0
        end_i = max(start_i + size, start_i)
        # end_i = min(center + half_size + (size % 2), ori_size[i])
        start.append(start_i)
        end.append(end_i)

    return start, end


def cropped2ori(crop_data, ori_size, isocenter, trans_in_size):
    """
    crop_data: the cropped data
    ori_size: the original size of the data
    isocenter: the isocenter of the original data
    trans_in_size: the in_size parameter in the transfromation of loader
    """

    assert (np.array(trans_in_size) == np.array(crop_data.shape)).all()

    start_coords, end_coords = offset_spatial_crop(
        roi_center=isocenter, roi_size=trans_in_size
    )

    # remove the padding
    crop_start, crop_end = [], []
    for i in range(len(ori_size)):
        if end_coords[i] > ori_size[i]:
            diff = end_coords[i] - ori_size[i]
            crop_start.append(diff // 2)
            crop_end.append(crop_data.shape[i] - diff + diff // 2)
        else:
            crop_start.append(0)
            crop_end.append(crop_data.shape[i])

    crop_data = crop_data[
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]

    pad_out = np.zeros(ori_size)

    pad_out[
        start_coords[0] : end_coords[0],
        start_coords[1] : end_coords[1],
        start_coords[2] : end_coords[2],
    ] = crop_data

    return pad_out
