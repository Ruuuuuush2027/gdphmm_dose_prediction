import torch
from data_loader_infer import GetLoader
from modeling.mednext_d2 import create_mednext_d2

import yaml
import os
import numpy as np
import argparse

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


  

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='GDP inference')
    parser.add_argument('cfig_path',  type = str)
    parser.add_argument('--phase', default = 'test', type = str)
    
    args = parser.parse_args()
    cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
    cfig["loader_params"]["val_bs"] = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    # ------------ data loader -----------------#
    loaders = GetLoader(cfig=cfig["loader_params"])
    
    current_data_loader = loaders.test_dataloader()
    
    
    print("{} patient(s) in the test set".format(len(current_data_loader.dataset)))
    # ------------ model -----------------#

    model_pth = cfig["save_model_path"]

    if not os.path.exists(model_pth):
        raise ValueError("The model path {} does not exist".format(model_pth))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if cfig["model_params"]["model_name"] == "mednext_d2":
        model = create_mednext_d2(including_optimizer=False)
    else:
        raise ValueError(
            "The model name {} is not supported".format(cfig["model_name"])
    )

  
    model.load_state_dict(
        torch.load(model_pth, map_location=device, weights_only=True)["model"]
    )
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(current_data_loader):
            
            assert len(data_dict["data"]) == 1, "The batch size should be 1"
            
            if data_dict["site"]== 1:
                outputs = model(data_dict["data"].to(device), task="HaN")
            elif data_dict["site"]== 2:
                outputs = model(data_dict["data"].to(device), task="Lung")
            else:
                raise ValueError("The site should be either 1 or 2")
            
            if cfig["act_sig"]:
                outputs = torch.sigmoid(outputs)
            outputs = outputs * cfig["scale_out"]

            if "label" in data_dict.keys():
                print(
                    "L1 error is ",
                    torch.nn.L1Loss()(outputs, data_dict["label"].to(device)).item(),
                )

            
            for index in range(len(outputs)):
                pad_out = np.zeros(
                    data_dict["ori_img_size"][index].numpy().tolist()
                )
                crop_data = outputs[index][0].cpu().numpy()
                ori_size = data_dict["ori_img_size"][index].numpy().tolist()
                isocenter = data_dict["ori_isocenter"][index].numpy().tolist()
                trans_in_size = cfig["loader_params"]["in_size"]
                pred2orisize = (
                    cropped2ori(crop_data, ori_size, isocenter, trans_in_size)
                    * cfig["loader_params"]["dose_div_factor"]
                )
                pid = data_dict["id"][index]

                pred2orisize = np.array(pred2orisize, dtype=np.float32)
                np.save(os.path.join(cfig['save_pred_path'], data_dict['id'][index] + '_pred.npy'), pred2orisize)
    

        
