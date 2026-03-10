import torch
from models.linear_scheduler import LinearNoiseScheduler, NoiseSchedulerConfig

# class NoiseSchedulerConfig:
#     num_timesteps: int
#     beta_start: float
#     beta_end: float


def test_init():
    config = NoiseSchedulerConfig(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )

    scheduler = LinearNoiseScheduler(config)
    im = torch.zeros(1, 1, 28, 28)

    # Sample random noise
    noise = torch.randn_like(im)

    # Sample timestep
    t = torch.randint(0, config.num_timesteps, (im.shape[0],))
    t = 1000 * torch.ones_like(t)

    # Add noise to images according to timestep
    noisy_im = scheduler.add_noise(im, noise, t)

    print(torch.norm(noisy_im - im))

    assert torch.testing.assert_close(noisy_im, im)

    assert True
