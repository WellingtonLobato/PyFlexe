<p align="center">
  <img src=img/Flexe_logo.png>
</p>

# PyFlexe
Flexe is a new framework for simulation of Federated Learning (FL) in Connected and Autonomous Vehicle (CAVs). Its adaptable design allows for the implementation of a variety of FL schemes, including horizontal, vertical, and Federated Transfer Learning. Flexe and PyFlexe are free to download and use, built for customization, and allows for the realistic simulation of wireless networking and vehicle dynamics. The project was partitioned into two, one of which is responsible for vehicle-to-vehicle communication (Flexe) and the other for the construction and training of models (PyFlexe).

## Getting Started

We developed FLEXE to make it possible to implement and develop vehicular FL applications within the context of CAVs. It further simplifies the process of modeling specific Machine Learning (ML) and FL applications into environments suitable for CAVs. Specifically, we developed Flexe on top of the Veins network simulator to simulate the dynamics of communication between vehicles.

### Prerequisites
PyFlexe requires the following software to be installed 

- OMNeT++
- conan
- grpc
- TensorFlow
- PyTorch

### OMNeT++ (6 >=) installation (Flexe)
Please do follow the instructions from the official [OMNeT documentation](https://doc.omnetpp.org/omnetpp/InstallGuide.pdf)

### Conan installation (Flexe)
Please do follow the instructions from the official [conan documentation](https://docs.conan.io/en/latest/installation.html)

### GRPC installation
Please do follow the instructions from the official [GRPC documentation](https://grpc.io/docs/languages/python/quickstart/)

### TensorFlow installation (2.11.0 >=)
Please do follow the instructions from the official [TensorFlow documentation](https://www.tensorflow.org/install)

### PyTorch installation (1.13.1 >=)
Please do follow the instructions from the official [PyTorch documentation](https://pytorch.org/tutorials/beginner/basics/intro.html)

### Installing

In order to install the necessary packages to run PyFlexe, just run the following command in the root directory.

```
poetry install
```


## Running the server

```
python server_flexe.py --ip 127.0.0.1 --dataset MNIST
```


## Project structure - main components 

	├── core
	│   ├── application
	│   │   ├── tf
	│   │   └── torch
	│   ├── dataset
	│   ├── model
	│   ├── proto
	│   └── server
	│       ├── common
	├── data
	│   ├── CIFAR10
	│   ├── MNIST
	│   ├── motion_sense
	│   └── UCI-HAR
	├── doc
	├── download.sh
	├── img
	│   └── Flexe_logo.png
	└── server_flexe.py


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

## Authors

* **Wellington Lobato** - [WellingtonLobato](https://github.com/WellingtonLobato)
* **Joahannes B. D. da Costa** - [joahannes](https://github.com/joahannes)
* **Allan M. de Souza** - [AllanMSouza](https://github.com/AllanMSouza)
* **Denis Rosario**
* **Christoph Sommer** - [sommer](https://github.com/sommer)
* **Leandro A. Villas**

# Citation

PyFlexe and Flexe can reproduce results in the following papers:

```tex
@INPROCEEDINGS{Lobato2022,
  author={Lobato, Wellington and Costa, Joahannes B. D. Da and Souza, Allan M. de and Rosário, Denis and Sommer, Christoph and Villas, Leandro A.},
  booktitle={2022 IEEE 96th Vehicular Technology Conference (VTC2022-Fall)}, 
  title={FLEXE: Investigating Federated Learning in Connected Autonomous Vehicle Simulations}, 
  year={2022},
  pages={1-5},
  doi={10.1109/VTC2022-Fall57202.2022.10012905}
}
```

## License

This project is licensed under the GPL-2.0 license - see the [COPYING.md](COPYING.md) file for details
