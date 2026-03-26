# Latent Diffusion

This an implementation of stable diffusion, meant for learning while also serving as a portfolio project. 
Latent diffusion trains a diffusion model in the latent space of a Variational Autoencoder, to enable 
training of diffusion models on large image sizes. Variations and improvements on this concept underpin 
many modern image generative models. 


In its current form, this implementation focusses on small datasets with the aim to be sufficiently 
flexible for extentions and ablations for improvements proposed in various papers. 

While Machine Learning is a fast moving field, this repository aims to be as rigorous as possible with 
the aim of providing self-documented code. That is to say that the modules have been written to be 
as readable and flexible as possible. 
All core parts of the model as well as all configurations have been unit tested. While already quite 
extensive. Extensive logging compatible with the standard logging has also been implemented as a 
feature and my considerations (as well as the design decisions) will be outlined later in a more 
extensive write up. 

During my PhD I have developed my own system to structure my machine learing code bases in a manner 
that is conceptually sound and reproducible across different projects, while providing sufficient 
flexibility and _without_ too much opions. In essence it is a set of design principles applied
in structured manner that form the bases, not a set of restricting rules.
The reader is invited to have a look at [DECT](https://github.com/ErnstRoell/dect-reimplementation)
for conceptually similar but easier project from a model implementation perspective. 

The structure between the projects is the same, but the tools can, may and will change. An 
underlying philosophy is the idea of high cohesion with low coupling of the code. Each of the 
independent modules can be excecuted independently with independent logging logic and minimal 
dependencies. It allows for easy development of the modules independently. 
The training and sampling scripts provide the glue that combines everything together. 


> [!NOTE]  
> I have not used any AI or LLM coding in _any_ part of this repository. All work is entirely my own.




## Installation and training.

For the installation we use `uv`, a modern package manager and PyTorch with Cuda enabled. 

## Usage


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

If you find our work useful, please consider using the following citation:
