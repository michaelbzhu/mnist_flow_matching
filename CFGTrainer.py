from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from common import ConditionalVectorField
from gaussian_probability_path import GaussianConditionalProbabilityPath

MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.losses = []

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs
    ) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(
            f"Training model with size: {size_b / MiB:.3f} MiB and class {type(self.model).__name__}"
        )

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            self.losses.append(loss.item())
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {idx}, loss: {loss.item():.3f}")

        # Finish
        self.model.eval()


class CFGTrainer(Trainer):
    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        model: ConditionalVectorField,
        eta: float,
        **kwargs,
    ):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size)  # (bs, c, h, w), (bs,1)

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0

        # Step 3: Sample t and x
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        alpha_t = self.path.alpha(t)
        beta_t  = self.path.beta(t)
        eps = torch.randn_like(z)
        x_t = alpha_t * z + beta_t * eps

        # Step 4: Regress and output loss
        predicted_error = self.model(x_t, t, y)  # (bs, 1, 32, 32)
        # loss = predicted_error - eps
        loss = torch.mean((predicted_error - eps).pow(2))
        return loss
