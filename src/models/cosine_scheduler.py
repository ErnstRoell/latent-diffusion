from dataclasses import dataclass

import torch


@dataclass
class SchedulerConfig:
    num_timesteps: int
    zero_snr: bool
    cosine_s: float


def make_cosine_schedule(n_timestep, cosine_s):
    timesteps = torch.arange(n_timestep + 1) / n_timestep + cosine_s
    alphas = timesteps / (1 + cosine_s) * torch.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = betas.clamp(max=0.999)
    return betas


def enforce_zero_terminal_snr(betas):
    """
    Rescaling the betas following the paper "Common Diffusion Noise Schedules
    and Sample Steps are Flawed" [1].

    Ensures the betas start at 1 and end at 0.

    [1] https://arxiv.org/pdf/2305.08891
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


class CosineNoiseScheduler(torch.nn.Module):
    def __init__(self, config: SchedulerConfig):
        super().__init__()
        self.config = config
        self.betas = make_cosine_schedule(config.num_timesteps, config.cosine_s)

        if config.zero_snr:
            self.betas = enforce_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod).reshape(-1, 1, 1, 1)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(
            1 - self.alpha_cum_prod
        ).reshape(-1, 1, 1, 1)

    def add_noise(self, im, noise, t):
        # Assumes im is of shape B,C,H,W
        return (
            self.sqrt_alpha_cum_prod[t] * im
            + self.sqrt_one_minus_alpha_cum_prod[t] * noise
        )

    def sample_timestep(self, im):
        # Sample timestep
        return torch.randint(
            0,
            self.config.num_timesteps,
            (im.shape[0],),
        )

    def sample_noise(self, im):
        return torch.randn_like(im)

    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / torch.sqrt(
            self.alpha_cum_prod[t]
        )
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = xt - ((self.betas[t]) * noise_pred) / (
            self.sqrt_one_minus_alpha_cum_prod[t]
        )
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape)
            return mean + sigma * z, x0
