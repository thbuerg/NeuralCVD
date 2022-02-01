<div align="center">    

![Logo](./src/neuralcvd_logo.png?raw=true "Logo")

**Neural network-based integration of polygenic and clinical information: Development and validation of a prediction model for 10 year risk of major adverse cardiac events in the UK Biobank cohort**

[![Paper](https://img.shields.io/badge/TheLancet-DigitalHealth-informational)](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00249-1/fulltext)

</div>
 
## Description   
Code related to the paper "Neural network-based integration of polygenic and clinical information: Development and validation of a prediction model for 10 year risk of major adverse cardiac events in the UK Biobank cohort". 
This repo is a python package for preprocessing UK Biobank data and preprocessing, training and evaluating the NeuralCVD score.

![NeuralCVD](./src/neuralcvd_fig1.png?raw=true "NeuralCVD")

## Methods
**NeuralCVD** is based on the fantastic [Deep Survival Machines](https://arxiv.org/abs/2003.01176) Paper, the original implementation can be found [here](https://github.com/autonlab/DeepSurvivalMachines).

## Assets
This repo contains code to preprocess [UK Biobank](https://www.ukbiobank.ac.uk/) data, train the NeuralCVD score and analyze/evaluate its performance.

- Preprocessing involves: parsing primary care records for desired diagnosis, aggregating the cardiovascular risk factors analyzed in the study and calculating predefined polygenic risk scores.
- Training involves Model specification via pytorch-lightning and hydra.
- Postprocessing involve extensive benchmarks with linear Models, and calculation of bootstrapped metrics.


## How to train the NeuralCVD Model  
1. First, install dependencies   
```bash
# clone project   
git clone https://github.com/thbuerg/NeuralCVD

# install project   
cd NeuralCVD
pip install -e .   
pip install -r requirements.txt
 ```   

2. Download UK Biobank data. Execute preprocessing notebooks on the downloaded data.

3. Edit the `.yaml` config files in `neuralcvd/experiments/config/`:
```yaml
setup:
  project_name: <YourNeptuneSpace>/<YourProject>
  root_dir: absolute/path/to/this/repo/
experiment:
  tabular_filepath: path/to/processed/data
```

4. Set up [Neptune.ai](https://www.neptune.ai)

5. Train the NeuralCVD Model (make sure you are on a machine w/ GPU)
 ```bash
# module folder
cd neuralcvd

# run training
bash experiments/run_NeuralCVD_S.sh
```

## Citation   
```
@article{steinfeldt2022neural,
  title={Neural network-based integration of polygenic and clinical information: development and validation of a prediction model for 10-year risk of major adverse cardiac events in the UK Biobank cohort},
  author={Steinfeldt, Jakob and Buergel, Thore and Loock, Lukas and Kittner, Paul and Ruyoga, Greg and zu Belzen, Julius Upmeier and Sasse, Simon and Strangalies, Henrik and Christmann, Lara and Hollmann, Noah and others},
  journal={The Lancet Digital Health},
  volume={4},
  number={2},
  pages={e84--e94},
  year={2022},
  publisher={Elsevier}
}
```  
