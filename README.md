# Resampled linear discriminant analysis
This repository contains the implementations to recreate the results of my bachelor thesis. Most of the implementation is based on the code from [1].

[1] J. Sosulski, J. Kemmer, and M. Tangermann, “Improving Covariance Matrices Derived from Tiny Training Datasets for the Classification of Event-Related Potentials with Linear Discriminant Analysis”, Neuroinformatics, pp. 1–16, 2020.

## Reproducing the results
To use this code and reproduce the results on Windows, you can clone the repository to your own machine. The Python version used is Python 3.8.8, and you can download specific versions of Python [here](https://www.python.org/downloads/). Make sure you have a version of Anaconda installed as well, I used Anaconda3. To create the environment, open up an Anaconda prompt, cd into the directory where you cloned the repository and run the following command:
```
conda env create -f environment.yml
```
If you want to create the environment differently, the libraries I have used can be found in 'requirements.txt'. After setting up the environment, modify the `local_config.yaml` file such that the root to the directory where the SPOT data is stored and the directory where you want the results to be stored on your own machine are in there.

Finally, to run the pipelines, you can open an Anaconda prompt and activate the environment,
```
conda activate 'ResampledLDA'
```
cd to the directory of the repository, and run the following command to run the resampled LDA pipeline on all the data:
 ```
 python Main_pipeline.py spot_single
 ```
Note that this __only__ runs the resampled LDA pipeline on the __whole__ SPOT dataset. If you want to run a different pipeline, you can go into the `Utils.py` file and in the function `create_lda_pipelines()` modify the variable in the classifier dictionary from ResLDA() to SLDA(), to run the shrinkage LDA.

If you only want to run the chosen pipeline on one subject, you can modify the command in the Anaconda prompt:
 ```
 python Main_pipeline.py spot_single 0
 ```
This means it will only run on the first subject. If you want to run it on another of the 13 subjects, change the 0 to another number in the [0-12] range. Finally, if you want to run the analysis on a different dataset, there are a number of valid datasets from the MOABB that you can run the analysis on. To do this, in the Anaconda command change `spot_single` to any of the following datasets: epfl, bnci_1, bnci_als, bnci_2, braininvaders. For example:
 ```
 python Main_pipeline.py bnci_1
 ```
