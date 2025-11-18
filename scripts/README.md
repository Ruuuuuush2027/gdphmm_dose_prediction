# gdphmm_dose_prediction

## Precomputing processed NPZ files

The training loader (`dataloading/efficient_data_loader_v2.py`) expects each `.npz` to already include the combined PTV/OAR channels and normalized image/dose arrays. Use `scripts/precompute_combined_npz.py` to materialize those attributes once so that training no longer retries the expensive combination logic on the fly.

Example command from the repository root:

```bash
python scripts/precompute_combined_npz.py \
	--datasets-root datasets \
	--sites HaN Lung \
	--split train \
	--output-suffix train_process \
	--scale-dose-dict meta_files/PTV_DICT.json \
	--pat-obj-dict meta_files/Pat_Obj_DICT.json \
	--skip-existing
```

This scans `datasets/<Site>/train`, generates the combined tensors (matching the logic in `data_loader_infer.py`), and writes results to `datasets/<Site>/train_process`. Point the CSV entries in `meta_files/meta_data.csv` (or set `Dataset` env variable) to the `_process` folders when training with the efficient loader.
