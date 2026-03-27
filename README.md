# Latent Diffusion

This an implementation of stable diffusion, meant for learning while also
serving as a portfolio project. Latent diffusion trains a diffusion model in
the latent space of a Variational Autoencoder, originally proposed to enable
training of diffusion models on large image sizes. Variations and improvements
on this concept underpin many modern image generative models. 

This repository aims to be as rigorous as possible. The code has been structured
to be both general, flexible and as decoupled as possible.
All core parts of the model as well as all configurations have been unit
tested. 

Logging compatible with the standard logging module has been implemented as a
feature and my considerations (as well as the design decisions) will be
outlined later in a more extensive write up. 

During my PhD I have developed my own system for structuring machine learning
code bases in a manner that is both conceptually sound and reproducible across
different projects, while providing sufficient flexibility _without_ too
many constraints.
In essence it is a set of design principles applied in structured manner that
form the bases, not a set of restricting rules. The reader is invited to have a
look at
[DECT](https://github.com/ErnstRoell/dect-reimplementation) for conceptually
similar but easier project from a model implementation perspective. 

The structure between the projects is the same, but the tools can, may and will
change. An underlying philosophy is the idea of high cohesion with low coupling
of the code. Each of the independent modules can be executed independently
with independent logging logic and minimal dependencies. It allows for easy
development of the modules independently. The training and sampling scripts
provide the glue that combines everything together. 

> [!NOTE]  
> No coding LLM has been used in _any_ part of this repository. All work is
> entirely my own.

> [!NOTE]  
> This repository is still under active development. More detailed
> documentation for the architecture and design decisions will be added
> gradually. 

## Installation and training.

For the installation we use `uv`, a modern package manager and PyTorch with Cuda enabled. 

## Usage

### Create the dataset

First create the dataset with the command 

```python
uv run src/datasets/mnist.py
```

The `main` function contains the setup code, which can be ran independently if required during testing. 

### Training the models

Each configuration file under `configs/` is a self-contained configuration and can be ran with the 
following commands. 

Train the diffusion model with 

```python
uv run src/train_ddpm.py --config configs/scratch/mnist_unet.yaml
```

Optionally add the `--dev` flag to run the model on the development dataset. 

Train the VAE with the command 

```python
uv run src/train_vae.py --config configs/scratch/mnist_vae.yaml
```

The above two command train a standard VAE and standard Diffusion model on the 
MNIST dataset. To train the latent diffusion model, a VAE model has to be trained 
first (in this case `mnist_vae.yaml`) and this configuration file needs to be 
passed in the latent diffusion model configuration. If the required artifacts 
exist, the latent model can be trained using the command

```python
uv run src/train_ddpm.py --config configs/scratch/mnist_latent_unet.yaml
```

### Sampling the DDPM

After training, samples can be generated using the sampler with the command 

```python
uv run src/sampler.py --config configs/scratch/mnist_unet.yaml
```

or 

```python
uv run src/sampler.py --config configs/scratch/mnist_latent_unet.yaml
```

### Running the unit tests

The unit tests can be ran with the command 

```
uv run pytest
```

In case a coverage report is desired, the tests can also be ran with 
the command `make tests` running the `Makefile` which will generate 
a coverage report. For the `DownBlock`, `MidBlock` and `UpBlock`, the 
coverage is relatively high. 


## License

Our code is released under a BSD-3-Clause license. This license essentially
permits you to freely use our code as desired, integrate it into your projects,
and much more --- provided you acknowledge the original authors. Please refer to
[LICENSE.md](LICENSE.md) for more information. 

## Contributing

We welcome contributions and suggestions for our DECT package! Here are some
basic guidelines for contributing:

### How to Submit an Issue

1. **Check Existing Issues**: Before submitting a new issue, please check if it
   has already been reported.

2. **Open a New Issue**: If your issue is new, open a new issue in the
   repository. Provide a clear and detailed description of the problem,
   including steps to reproduce the issue if applicable.

3. **Include Relevant Information**: Include any relevant information, such as
   system details, version numbers, and screenshots, to help us understand and
   resolve the issue more efficiently.

### How to Contribute

If you're unfamiliar with contributing to open source repositories, here is a
basic roadmap:

1. **Fork the Repository**: Start by forking the repository to your own GitHub
   account.

2. **Clone the Repository**: Clone the forked repository to your local machine.

   ```sh
   git clone https://github.com/your-username/dect.git
   ```

3. **Create a Branch**: Create a new branch for your feature or bug fix.

   ```sh
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Implement your changes in the new branch.

5. **Commit Changes**: Commit your changes with a descriptive commit message.

   ```sh
   git commit -m "Description of your changes"
   ```

6. **Push Changes**: Push the changes to your forked repository.

   ```sh
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**: Open a pull request to the main repository with a
   clear description of your changes and the purpose of the contribution.


### Need Help?

If you need any help or have questions, feel free to reach out to the authors or
submit a pull request. We appreciate your contributions and are happy to assist!


## Acknowledgement 

As an original starting point,
![this](https://github.com/explainingai-code/StableDiffusion-PyTorch)
repository has been used and it has been very helpful as an initial starting 
point.

