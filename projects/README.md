# Simulation of Data-Driven, Neuron-Type Specific CA3 SNNs 
This repository includes information as to how to run an example and full-scale spiking neural network (SNN) model of hippocampal subregion CA3. Before following the guidelines, please follow the instructions provided by CARLsim provided in their [README](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc).

## Installation
Beyond the dependencies of CARLsim4 described at the link above, to generate the syntax necessary to run the example and full-scale SNNs one will need to install Python 3 as well as the package dependencies included in the table below. Additionally, one can install the following [Anaconda distribution](https://docs.anaconda.com/anaconda/install/), which includes Python 3 and pandas, but the xlrd function will still need to be downloaded, as it is an optional dependency of pandas.

## Module Dependencies:

|module|tested version|
|---|---|
|Anaconda|02.2020|
|Python|3.7.6|
|pandas|0.25.3|
|numpy|1.18.1|
|xlrd|1.2.0|
|boost|1.67.0|
