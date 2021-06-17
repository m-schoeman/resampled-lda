# Resampled linear discriminant analysis
This repository contains the implementations to recreate the results of my bachelor thesis. Most of the implementation is based on the code from [1].

[1] J. Sosulski, J. Kemmer, and M. Tangermann, “Improving Covariance Matrices Derived from Tiny Training Datasets for the Classification of Event-Related Potentials with Linear Discriminant Analysis”, Neuroinformatics, pp. 1–16, 2020.

## Reproducing the results
To use this code and reproduce the results on Windows, you can clone the repository to your own machine. You need to create an evironment and then you can install the libraries that were used from requirements.txt like so: 
```
pip install -r /path/to/requirements.txt
```
After installing the requirements, modify the local_config.yaml file such that the roots to the directory where the SPOT data is stored and the directory where you want the results to be stored on your own machine are in there.\\
Finally, to run the pipelines, you can open an Anaconda prompt and activate the environment,
```
conda activate 'name of the environment'
```
cd to the directory in which the .py files to run the pipelines are located, and run the following command to run the resampled LDA pipeline on all the data:
 ```
 python main_pipeline.py spot_single
 ```
Note that this __only__ runs the resampled LDA pipeline on the __whole__ SPOT dataset. If you want to run a different pipeline, you can go into the Utils.py file and in the function create_lda_pipelines() modify the variable in the classifier dictionary from ResLDA() to SLDA(), to run the shrinkage LDA.

If you only want to run the chosen pipeline on one subject, you can modify the command in the Anaconda prompt:
 ```
 python main_pipeline.py spot_single 0
 ```
This means it will only run on the first subject. If you want to run it on another of the 13 subjects, change the 0 to another number in the [0-12] range. Finally, if you want to run the analysis on a different dataset, there are a number of valid datasets from the MOABB that you can run the analysis on. To do this, in the Anaconda command change spot_single to any of the following datasets: epfl, bnci_1, bnci_als, bnci_2, braininvaders. 
