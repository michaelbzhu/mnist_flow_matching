Goal: compare different architectures on generating images from the MNIST dataset using flow matching

take the scaffolding code from [MIT's Introduction to Flow Matching and Diffusion Models course](https://diffusion.csail.mit.edu/)
- credit to [Peter E. Holderrieth](https://www.peterholderrieth.com/) and [Ezra Erives](https://eerives.me/)

architectures to compare:
- UNet model (from the course) -  `unet.py`
- Diffusion Transformer - `dit.py`
- Mamba - `mamba.py`

code organization:
- `common.py` contains shared abstract classes `Sampleable` and `ConditionalVectorField`
- `gaussian_probability_path.py` contains code for `GaussianConditionalProbabilityPath` which is used to add noise to mnist images during training
- `simulator_utils.py` contains the definitions for `ODE` (ordinary differential equation) and the ODE simulator
- `mnist_sampler.py` contains the implementation of the MNIST dataloader / sampler
- `CFGTrainer.py` contains code for the `CFGTrainer` class

training run and example generations can be seen in `experiment.ipynb`
