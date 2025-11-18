export Dataset="/project2/ruishanl_1185/dose_prediction_mo/gdphmm_dose_prediction/datasets"
export Result="/project2/ruishanl_1185/dose_prediction_mo/gdphmm_dose_prediction/results"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CONDA_PREFIX=/home1/mojiang/.conda/envs/pytorch_cv
export PATH="/home1/mojiang/.conda/envs/pytorch_cv/bin:$PATH"

conda init
conda activate pytorch_cv

python train.py \
    -p test_project_name \
    -c train.yaml \
    --fp 16