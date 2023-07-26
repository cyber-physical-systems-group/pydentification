# pydentification

Toolbox for dynamical system identification with neural networks. It includes mostly neural network models, which are 
created during research of CPSG, but also other models, such as reimplementations of models from literature and various
other benchmarks. It will also include utilities for training, testing and evaluation of models.

# Usage

## Installation

Right now, this is internal library (with intents for making it public soon), so just clone and run it. 

## Publishing

We will solve this when we get there...

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
Short form is enough, like `[DynoNet](feat) Add reimplementation of dynamica module from DynoNet paper` or
`[General](chore) Update requirements.txt`.

## Code Style

Python code style using `black`, `flake8` and `isort` is enforced by GitHub Actions. Typing is not required, but checked
using `mypy`. Code style and testing requirements can be installed using `pip install -r requirements-dev.txt`.

### Testing

Some unittests are included, but not required, as large parts of the repository will be research-only. It is usually a
good idea to have some tests for the code. Tests for existing cases are run on GitHub Actions, so braking changes will
be detected.