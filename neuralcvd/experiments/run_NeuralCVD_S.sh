#! /bin/sh
#BSUB -q graphical
#BSUB -R "rusage[mem=30000]"
#BSUB -M 30000
#BSUB -W 100:00
#BSUB -n 12
echo Starting.
cd ../../

echo $(hostname)
echo $(which python)
echo $(python -c 'import torch; print(f"found {torch.cuda.device_count()} gpus.")')
echo $CUDA_VISIBLE_DEVICES

python neuralcvd/experiments/train_survival.py --config-dir neuralcvd/experiments/config/ --config-name CVD_S_CORE &
python neuralcvd/experiments/train_survival.py --config-dir neuralcvd/experiments/config/ --config-name CVD_S_PGS

echo Done with submission script
