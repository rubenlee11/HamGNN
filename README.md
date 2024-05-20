<p align="center">
  <img height="130" src="logo/logo.png"/>
</p>

## Contents
- [Introduction to HamGNN](#introduction-to-hamgnn)
- [Requirements](#requirements)
  - [Python libraries](#python-libraries)
  - [OpenMX](#openmx)
  - [openmx_postprocess](#openmx_postprocess)
  - [read_openmx](#read_openmx)
- [Installation](#installation)
- [Usage](#usage)
  - [Preparation of Hamiltonian Training Data](#preparation-of-hamiltonian-training-data)
  - [Graph Data Conversion](#graph-data-conversion)
  - [HamGNN Network Training and Prediction](#hamgnn-network-training-and-prediction)
  - [Details of training for bands (The 2nd training/fine-tuning step)](#details-of-training-for-bands-the-2nd-trainingfine-tuning-step)
  - [Band Structure Calculation](#band-structure-calculation)
- [The support for ABACUS software](#the-support-for-abacus-software)
- [Diagonalizing Hamiltonian matrices for large scale systems](#diagonalizing-hamiltonian-matrices-for-large-scale-systems)
  - [Installation](#installation-1)
  - [Usage](#usage-1)
- [Explanation of the parameters in config.yaml](#explanation-of-the-parameters-in-configyaml)
- [References](#references)
- [Code contributors](#code-contributors)
- [Project leaders](#project-leaders)

## Introduction to HamGNN
The HamGNN model is an E(3) equivariant graph neural network designed for the purpose of training and predicting tight-binding (TB) Hamiltonians of molecules and solids. Currently, HamGNN can be used in common ab initio DFT software that is based on numerical atomic orbitals, such as OpenMX, Siesta, and Abacus. HamGNN supports predictions of SU(2) equivariant Hamiltonians with spin-orbit coupling effects. HamGNN not only achieves a high fidelity approximation of DFT but also enables transferable predictions across material structures, making it suitable for high-throughput electronic structure calculations and accelerating computations on large-scale systems.

## Requirements

The following environments and packages are required to use HamGNN:

### Python libraries
We recommend using the Python 3.9 interpreter. HamGNN needs the following python libraries:
- numpy == 1.21.2
- PyTorch == 1.11.0
- PyTorch Geometric == 2.0.4
- pytorch_lightning == 1.5.10
- e3nn == 0.5.0
- pymatgen == 2022.3.7
- tensorboard == 2.8.0
- tqdm
- scipy == 1.7.3
- yaml

A convenient way to set up the Python environment for HamGNN is to use the HamGNN conda environment I have uploaded to this [website](https://zenodo.org/records/11064223). Users can simply extract this conda environment directly to their own `conda/envs` directory.
### OpenMX
HamGNN aims to fit the TB Hamiltonian generated by `OpenMX`. The user need to know the basic OpenMX parameters and how to use them properly. OpenMX can be downloaded from this [site](https://www.openmx-square.org/).

### openmx_postprocess
openmx_postprocess is a modified OpenMX package used for computing overlap matrices and other Hamiltonian matrices that can be calculated analytically. The data computed by openmx_postprocess will be stored in a binary file `overlap.scfout`. The installation and usage of openmx_postprocess is essentially the same as that of OpenMX. To install openmx_postprocess, you need to install the [GSL](https://www.gnu.org/software/gsl/) library first.Then enter the openmx_postprocess directory and modify the following parameters in the makefile:
+ `GSL_lib`: The lib path of GSL
+ `GSL_include`: The include path of GSL
+ `MKLROOT`: The intel MKL path
+ `CMPLR_ROOT`: The path where the intel compiler is installed

After modifying the makefile, you can directly execute the make command to generate two executable programs, `openmx_postprocess` and `read_openmx`.

### read_openmx
read_openmx is a binary executable that can be used to export the matrices from the binary file overlap.scfout to a file called `HS.json`.

## Installation
Run the following command to install HamGNN:
```bash
git clone https://github.com/QuantumLab-ZY/HamGNN.git
cd HamGNN
python setup.py install
```

## Usage
### Preparation of Hamiltonian Training Data
First, generate a set of structure files (POSCAR or CIF files) using molecular dynamics or random perturbation. After setting the appropriate path parameters in the `poscar2openmx.py` file,
run `python poscar2openmx.py` to convert these structures into OpenMX's `.dat` file format. Run OpenMX to perform static calculations on these structure files and obtain the `.scfout` binary files, which store the Hamiltonian and overlap matrix information for each structure. These files serve as the target Hamiltonians during training. Next, run `openmx_postprocess` to process each structure and obtain the `overlap.scfout` file, which contains the Hamiltonian matrix H0 that is independent of the self-consistent charge density. If the constructed dataset is only used for prediction purposes and not for training (i.e., no target Hamiltonian is needed), run `openmx_postprocess` to obtain the `overlap.scfout` file merely. `openmx_postprocess` is executed similarly to OpenMX and supports MPI parallelism.

### Graph Data Conversion
After setting the appropriate path information in a `graph_data_gen.yaml` file, run `graph_data_gen --config graph_data_gen.yaml` to package the structural information and Hamiltonian data from all `.scfout` files into a single `graph_data.npz` file, which serves as the input data for the HamGNN network.

### HamGNN Network Training and Prediction
Prepare the `config.yaml` configuration file and set the network parameters, training parameters, and other details in this file. To run HamGNN, simply enter `HamGNN --config config.yaml`. Running `tensorboard --logdir train_dir` allows real-time monitoring of the training progress, where `train_dir` is the folder where HamGNN saves the training data, corresponding to the `train_dir` parameter in `config.yaml`. To enhance the transferability and prediction accuracy of the network, the training is divided into two steps. The first step involves training with only the loss value of the Hamiltonian in the loss function until the Hamiltonian training converges or the error reaches around 10^-5 Hartree, at which point the training can be stopped. Then, the band energy error is added to the loss function, and the network parameters obtained from the previous step are loaded for further training. After obtaining the final network parameters, the network can be used for prediction. First, convert the structures to be predicted into the input data format (`graph_data.npz`) for the network, following similar steps and procedures as preparing the training set. Then, in the `config.yaml` file, set the `checkpoint_path` to the path of the network parameter file and set the `stage` parameter to `test`. After configuring the parameters in `config.yaml`, running `HamGNN --config config.yaml` will perform the prediction. 
Several pre-trained models and the `config.yaml` file for the test examples are available on Zenodo (https://doi.org/10.5281/zenodo.8147631).

### Details of training for bands (The 2nd training/fine-tuning step)
When the training of the Hamiltonian matrix is completed in the first step, it is necessary to use the trained network weights to initialize the HamGNN network and start training for the energy bands. The parameters related to energy band training are as follows:
+ `checkpoint_path` parameter should be set to path of the weight file obtained after training on the Hamiltonian matrix in the first step.
+ Set `load_from_checkpoint` to True
+ `lr` should not be too large, it is recommended to use 0.0001.
+ In `losses_metrics` and `metrics`, remove the commented section for `band_energy`.
+ Set `calculate_band_energy` to True and specify the parameters `num_k`, `band_num`, and `k_path`.

After setting the above parameters, start the training again.

### Band Structure Calculation
Set the parameters in band_cal.yaml, mainly the path to the Hamiltonian data, then run `band_cal --config band_cal.yaml`

## The support for ABACUS software
The utilities to support ABACUS software have been uploaded in the `utils_abacus` directory. Users need to modify the parameters in the scripts within this directory. The code for `abacus_postprocess` in `utils_abacus/abacus_H0_export` is derived from modifying the `abacus` program based on ABACUS-3.5.3. The function of this tool is similar to `openmx_postprocess` and it is used to export the Hamiltonian part `H0`, which is independent of the self-consistent field (SCF) charge density. Compilation of `abacus-postprocess` is the same as that of the original ABACUS.

`poscar2abacus.py` and `graph_data_gen_abacus.py` scripts are respectively utilized for generating ABACUS structure files and packaging the Hamiltonian matrix into the `graph_data.npz` file. Users can explore the usage of these tools independently. Later on, I'll briefly introduce the meanings of the parameters within these scripts.

## Diagonalizing Hamiltonian matrices for large scale systems
For crystal structures containing thousands of atoms, diagonalizing the Hamiltonian matrix using the serial `band_cal` script can be quite challenging. To address this, we've introduced a multi-core parallel `band_cal_parallel` script within band_cal_parallel directory. Note: In certain MKL environments, using the `band_cal_parallel` may trigger a bug that reports the error message 'Intel MKL FATAL ERROR: Cannot load symbol MKLMPI_Get_wrappers'. Users can try the solutions provided in Issues [#18](https://github.com/QuantumLab-ZY/HamGNN/issues/18) and [#12](https://github.com/QuantumLab-ZY/HamGNN/issues/12) to resolve this issue (thanks to the help from `flamingoXu` and `newplay`).

### Installation
pip install mpitool-0.0.1-cp39-cp39-manylinux1_x86_64.whl

pip install band_cal_parallel-0.1.12-py3-none-any.whl

### Usage
In the Python environment with `band_cal_parallel` installed, execute the following command with multiple cpus to compute the band structure:
mpirun -np ncpus band_cal_parallel --config band_cal_parallel.yaml

##  Explanation of the parameters in config.yaml
The input parameters in config.yaml are divided into different modules, which mainly include `'setup'`, `'dataset_params'`, `'losses_metrics'`, `'optim_params'` and network-related parameters (`'HamGNN_pre'` and `'HamGNN_out'`). Most of the parameters work well using the default values. The following introduces some commonly used parameters in each module.
+ `setup`:
    + `stage`: Select the state of the network: training (`fit`) or testing (`test`).
    + `property`：Select the type of physical quantity to be output by the network, generally set to `hamiltonian`
    + `num_gpus`: number of gpus to train on (`int`) or which GPUs to train on (`list` or `str`) applied per node.
    + `resume`: resume training (`true`) or start from scratch (`false`).
    + `checkpoint_path`: Path of the checkpoint from which training is resumed (`stage` = `fit`) or path to the checkpoint you wish to test (`stage` = `test`).
    + `load_from_checkpoint`: If set to `true`, the model will be initialized with weights from the checkpoint_path.
+ `dataset_params`:
    + `graph_data_path`: The directory where the processed compressed graph data files (`grah_data.npz`) are stored.
    + `batch_size`: The number of samples or data points that are processed together in a single forward and backward pass during the training of a neural network. defaut: 1. 
    + `train_ratio`: The proportion of the training samples in the entire data set.
    + `val_ratio`: The proportion of the validation samples in the entire data set.
    + `test_ratio`：The proportion of the test samples in the entire data set.
+ `losses_metrics`：
    + `losses`: define multiple loss functions and their respective weights in the total loss value. Currently, HamGNN supports `mse`, `mae`, and `rmse`. 
    + `metrics`：A variety of metric functions can be defined to evaluate the accuracy of the model on the validation set and test set.
+ `optim_params`：
    + `min_epochs`: Force training for at least these many epochs.
    + `max_epochs`: Stop training once this number of epochs is reached.
    + `lr`：learning rate, the default value is 0.001.

+ `profiler_params`:
    + `train_dir`: The folder for saving training information and prediction results. This directory can be read by tensorboard to monitor the training process.

+ `HamGNN_pre`: The representation network to generate the node and pair interaction features
    + `num_types`：The maximum number of atomic types used to build the one-hot vectors for atoms
    + `cutoff`: The cutoff radius adopted in the envelope function for interatomic distances.
    + `cutoff_func`: which envelope function is used for interatomic distances. Options: `cos` refers to cosine envelope function, `pol` refers to the polynomial envelope function.
    + `rbf_func`: The radial basis function type used to expand the interatomic distances
    + `num_radial`: The number of Bessel basis.
    + `num_interaction_layers`: The number of interaction layers or orbital convolution layers.
    + `add_edge_tp`: Whether to utilize the tensor product of i and j to construct pair interaction features. This option requires a significant amount of memory, but it can sometimes improve accuracy.
    + `irreps_edge_sh`: Spherical harmonic representation of the orientation of an edge
    + `irreps_node_features`: O(3) irreducible representations of the initial atomic features
    + `irreps_edge_output`: O(3) irreducible representations of the edge features to output
    + `irreps_node_output`: O(3) irreducible representations of the atomic features to output
    + `feature_irreps_hidden`: intermediate O(3) irreducible representations of the atomic features in convelution
    + `irreps_triplet_output(deprecated)`: O(3) irreducible representations of the triplet features to output
    + `invariant_layers`: The layers of the MLP used to map the invariant edge embeddings to the weights of each tensor product path
    + `invariant_neurons`: The number of the neurons of the MLP used to map the invariant edge embeddings to the weights of each tensor product path

+ `HamGNN_out`: The output layer to transform the representation of crystals into Hamiltonian matrix
    + `nao_max`: It is modified according to the maximum number of atomic orbitals in the data set, which can be `14`, `19`, `26`.For short-period elements such as C, Si, O, etc., a nao_max of 14 is sufficient; the number of atomic bases for most common elements does not exceed 19. Setting nao_max to 26 would allow the description of all elements supported by OpenMX. For the Hamiltonian of ABACUS, `nao_max` can be set to either `27` (without Al, Hf, Ta, W) or `40` (supporting all elements in ABACUS).
    + `add_H0`: Generally true, the complete Hamiltonian is predicted as the sum of H_scf plus H_nonscf (H0)
    + `symmetrize`：if set to true, the Hermitian symmetry constraint is imposed on the Hamiltonian
    + `calculate_band_energy`: Whether to calculate the energy bands to train the model 
    + `num_k`: When calculating the energy bands, the number of K points to use
    + `band_num_control`: `dict`: controls how many orbitals are considered for each atom in energy bands; `int`: [vbm-num, vbm+num]; `null`: all bands
    + `k_path`: `auto`: Automatically determine the k-point path; `null`: random k-point path; `list`: list of k-point paths provided by the user
    + `soc_switch`: if true, Fit the SOC Hamiltonian
    + `nonlinearity_type`: `norm` activation or `gate` activation as the nonlinear activation function

## References

The papers related to HamGNN:

[[1] Transferable equivariant graph neural networks for the Hamiltonians of molecules and solids](https://doi.org/10.1038/s41524-023-01130-4)

[[2] Universal Machine Learning Kohn-Sham Hamiltonian for Materials](https://arxiv.org/abs/2402.09251)

[[3] Accelerating the electronic-structure calculation of magnetic systems by equivariant neural networks](https://arxiv.org/abs/2306.01558)

[[4] Topological interfacial states in ferroelectric domain walls of two-dimensional bismuth](https://arxiv.org/abs/2308.04633)

[[5] Transferable Machine Learning Approach for Predicting Electronic Structures of Charged Defects](https://arxiv.org/abs/2306.08017)

## Code contributors:
+ Yang Zhong (Fudan University)
+ Changwei Zhang (Fudan University)
+ Zhenxing Dai (Fudan University)
+ Yuxing Ma (Fudan University)

## Project leaders: 
+ Hongjun Xiang  (Fudan University)
+ Xingao Gong  (Fudan University)

