import pydantic
import torch


class NoiseSchedulerConfig(pydantic.BaseModel):
    module: str
    num_timesteps: int
    beta_start: float
    beta_end: float
    zero_snr: bool


def make_cosine_schedule(n_timestep, cosine_s=8e-3):
    timesteps = torch.arange(n_timestep + 1) / n_timestep + cosine_s
    alphas = timesteps / (1 + cosine_s) * torch.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = betas.clamp(max=0.999)
    return betas


def make_linear_schedule(num_timesteps, linear_start=1e-4, linear_end=2e-2):
    return torch.linspace(linear_start, linear_end, num_timesteps)


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


class LinearNoiseScheduler(torch.nn.Module):
    def __init__(self, config: NoiseSchedulerConfig):
        super().__init__()
        self.config = config
        betas = (
            torch.linspace(
                config.beta_start**0.5,
                config.beta_end**0.5,
                config.num_timesteps,
            )
            ** 2
        )
        if config.zero_snr:
            self.betas = enforce_zero_terminal_snr(betas)
        else:
            self.betas = betas

        alphas = 1.0 - self.betas
        alpha_cum_prod = torch.cumprod(alphas, dim=0)
        sqrt_alpha_cum_prod = torch.sqrt(alpha_cum_prod).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - alpha_cum_prod).reshape(
            -1, 1, 1, 1
        )
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cum_prod", alpha_cum_prod)
        self.register_buffer("sqrt_alpha_cum_prod", sqrt_alpha_cum_prod)
        self.register_buffer(
            "sqrt_one_minus_alpha_cum_prod",
            sqrt_one_minus_alpha_cum_prod,
        )

    def add_noise(self, im, noise, t):
        # Assumes im is of shape B,C,H,W
        return (
            self.sqrt_alpha_cum_prod[t] * im
            + self.sqrt_one_minus_alpha_cum_prod[t] * noise
        )

    def sample_timestep(self, im):
        # Sample timestep
        return torch.randint(
            0, self.config.num_timesteps, (im.shape[0],), device=im.device
        )

    def sample_noise(self, im):
        return torch.randn_like(im, device=im.device)

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
            z = torch.randn(xt.shape, device=xt.device)
            return mean + sigma * z, x0
