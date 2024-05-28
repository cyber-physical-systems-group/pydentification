# pydentification

![Python](https://img.shields.io/badge/python-3.11-3670A0?style=flat&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=flat)](./LICENSE)

Toolbox for dynamical system identification with neural networks. It includes mostly neural network models, which are 
created during research of CPSG, but also other models, such as reimplementations of models from literature and various
other benchmarks. It will also include utilities for training, testing and evaluation of models.

# Usage

## Installation

To install simply run command `pip install git+https://github.com/cyber-physical-systems-group/pydentification.git` and
selected the version using tag (see [releases](https://github.com/cyber-physical-systems-group/pydentification/releases)).
Alternatively, for using the latest version, you can use `pip install .` after cloning the repository.

This will install the package and all its base dependencies. Note, that some of the submodules require extra
dependencies. To install requirements for development, run `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`.
This will install all the requirements for running tests and code style checks, without `pydentification` itself.

Right now, `pydentification` is not available on PyPI, so installing using git is the only possibility.

## Publishing

`pydentification` is a research toolbox for working with neural networks and dynamical systems, most of the work is 
meant to be published as research paper, however many of the utilities are shared. When publishing, make sure to include
reference to the version of the package used to develop the experiments. When contributing to `pydentification`, make
sure You include git tag (and GitHub release) describing features introduced for some paper. Each publication should 
have a tractable version of this package, which was used to develop the experiments.

## Docker

Additionally, `containers` repository in this organization contains Dockerfiles for deploying docker with certain
requirements, which can be used to speed up development. Right now, `containers` repository is not public.

*Note*: Right now, to use CUDa it is required to install PyTorch separately, since it is not included in the base
requirements. Recommended way is to use it with PyTorch Docker image with NVIDIA drivers.

# Conventions

## Branching

Submit changes only via pull requests! There is no branch naming convention, since this is a research repository, but
please try to make them meaningful.

## Committing

### Commit Signing

Please use only signed commits, using GPG key connected to your GitHub account in the organization. To do this follow [
this guide](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits). 

This requires you to have GPG key, which you can generate using `gpg --full-generate-key` command, which is added to
your GitHub account. You can then list your keys using `gpg --list-secret-keys --keyid-format LONG` and copy the key ID. 
Then you can add it to your GitHub account using `gpg --armor --export KEY_ID` and copy the output to GitHub. 

Then you can configure git to use this key using `git config --global user.signingkey KEY_ID` and `git config --global
commit.gpgsign true`. Also, make sure the email you use for commits is the same as the one you use for your GitHub,
which can be set by running `git config --global user.email`.

### Commit Messages

Please try to use conventional commits, which are described in [this guide](https://www.conventionalcommits.org/en/v1.0.0/).
This is not strictly enforced, but it is a good practice to follow, since it makes it easier to understand the history.
Short form is enough, like `[DynoNet](feat) Add reimplementation of dynamical module from DynoNet paper` or
`[General](chore) Update requirements.txt`.

## Code Style

Python code style using `black`, `flake8` and `isort` is enforced by GitHub Actions. Typing is not required, but checked
using `mypy`. Code style and testing requirements can be installed using `pip install -r requirements-dev.txt`.

### Testing

Some unittests are included, but not required, as large parts of the repository will be research-only. It is usually a
good idea to have some tests for the code. Tests for existing cases are run on GitHub Actions, so braking changes will
be detected.
