# GenRank

**GenRank: Regionalization of Equifinal Parameter Vectors at Ungauged Basins** authored by Abhinav Gupta and Mukesh Kumar

<p align="center">
  <img src="figures/github_schematic.png" width="900">
</p>

This repository contains the code associated with the paper **"Regionalization of the equifinal parameter vectors at ungauged basins."**

The project develops neural-network-based approaches for regionalizing **equifinal parameter ensembles** of a conceptual hydrological model from gauged to ungauged basins. The repository also includes a complementary neural-network framework for predicting hydrological model performance from model parameters and basin attributes.

---

## Overview

The workflow consists of two related components:

1. **Equifinal parameter-vector regionalization**

   * Uses parameter ensembles generated through DREAM-based calibration.
   * Uses static basin attributes as predictors.
   * Learns the distribution of behavioral/equifinal parameter vectors.
   * Generates an ensemble of parameter vectors for an ungauged basin.

2. **Hydrological performance prediction**

   * Uses hydrological model parameters together with static basin attributes.
   * Predicts hydrological model performance metrics.
   * Provides a way to evaluate and rank generated parameter vectors before running the hydrological model.
---

## Hydrological Model Parameters

The framework can be applied to any hydrological model. Our study used the following model
* **PT (Priestley–Taylor)** evapotranspiration formulation
* **SNOW-17** snow model
* **SAC-SMA** soil-moisture accounting model
* **Routing model**

The parameter vector includes 26 model parameters used as neural-network outputs.

The implementation includes parameters such as:

* PT alpha
* SNOW-17 parameters including `SCF`, `PXTEMP`, `TTI`, `MFMAX`, `MFMIN`, `UADJ`, `TIPM`, `PLWHC`, `NMF`, and `DAYGM`
* SAC-SMA parameters including `uztwm`, `uzfwm`, `lztwm`, `lzfpm`, `lzfsm`, `uzk`, `lzpk`, `lzsk`, `zperc`, `rexp`, `pfree`, `adimp`, and `rserv`
* Routing parameters `alpha` and `beta`

The parameter bounds used by the implementation are defined directly in the training scripts.

---

## Basin Attributes

The neural networks use static basin characteristics derived from the **CAMELS** dataset.

The current implementation combines attributes describing:

* Climate
* Geology
* Soil
* Topography
* Vegetation

A total of **26 static basin attributes** were used a as predictors.

---

## Repository Structure

```text
GenRank/
│
├── NN/
│   ├── MLP_ensemble.py
│   └── MLP_metrics.py
│
├── main_RegPerform.py
├── main_RegPerform_hyperparameter_optimization.py
├── main_metrics.py
├── main_metrics_hyperparameter_optimization.py
│
├── LICENSE
└── README.md
```

### Main files

| File                                             | Description                                                                                                     |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `main_RegPerform.py`                             | Train and evaluate the neural network for equifinal parameter-vector regionalization                            |
| `main_RegPerform_hyperparameter_optimization.py` | Hyperparameter optimization for the parameter-regionalization model                                             |
| `main_metrics.py`                                | Train and evaluate the neural network for hydrological performance prediction                                   |
| `main_metrics_hyperparameter_optimization.py`    | Hyperparameter optimization for the performance-prediction model                                                |
| `NN/MLP_ensemble.py`                             | Neural-network architectures, conditional generator, training utilities, ensemble evaluation, and custom losses |
| `NN/MLP_metrics.py`                              | MLP architecture and training/evaluation utilities for performance-metric prediction                            |

---

## Data Requirements

The code expects a local project directory containing the required input data and model results.

The exact location of these directories is controlled by the `main_dir` variable in the scripts.

**Before running the code, update `main_dir` to the location of your local project directory.**

For example:

```python
main_dir = 'D:/Research/Regionalization'
```

should be changed to the appropriate path on your system.

---

## DREAM Parameter Ensembles

The regionalization workflow uses parameter ensembles generated from DREAM-based calibration.

For each basin, the parameter files contain:

* Hydrological model parameters
* NSE
* NSAE
* KGE
* Additional calibration/output information

The implementation can identify high-performing parameter vectors using combinations of these performance metrics.

The current workflow computes a combined normalized score using NSE, NSAE, and KGE and uses the resulting scores to identify high-performing parameter vectors.

---
## Running the Code

### 1. Parameter-vector regionalization

After preparing the input data and updating `main_dir`, run:

```bash
python main_RegPerform.py
```

This trains the neural network and evaluates its ability to generate parameter ensembles.

---

### 2. Hyperparameter optimization

To perform hyperparameter optimization for the parameter-regionalization model:

```bash
python main_RegPerform_hyperparameter_optimization.py
```

The hyperparameter search space can be modified within the script.

---

### 3. Performance prediction

To train the performance-prediction network:

```bash
python main_metrics.py
```

---

### 4. Performance-model hyperparameter optimization

```bash
python main_metrics_hyperparameter_optimization.py
```

---

## Citation
Coming soon

---

## Author

**Abhinav Gupta**

Hydrological modeling
Neural networks · Hydrological modeling · Regionalization · Ungauged basins

GitHub: [AbhinavGupta1611](https://github.com/AbhinavGupta1611)

---

## Acknowledgments

This work builds on conceptual hydrological modeling, DREAM-based parameter estimation, CAMELS basin attributes, and neural-network methods for hydrological model regionalization.
