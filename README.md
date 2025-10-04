# MNIST Flow Matching with Different Architectures

Goal: compare different architectures on generating images from the MNIST dataset using flow matching

scaffolding/utility code from [MIT's Introduction to Flow Matching and Diffusion Models course](https://diffusion.csail.mit.edu/)
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

## Results
training run and example generations can be seen in `experiment.ipynb`
- upon visual inspection, the mamba generated samples are noticably worse than the other models, they have more artifacts and the digits are less clear
- DiT and UNet seem pretty comparable in generation quality
- interesting to note that the DiT (16.8M params) has 14x more parameters than the UNet (1.2M params) but training speed is only 5% slower

avg loss of final 500 steps:
- UNet: 131.15
- DiT: 123.94
- Mamba: 154.31

the avg loss values are pretty consistent when aggregating over final 1000, 500, 100, and 50 steps
