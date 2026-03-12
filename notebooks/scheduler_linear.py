%load_ext autoreload    
%autoreload 2
from models.linear_scheduler import LinearNoiseScheduler, NoiseSchedulerConfig
from models.cosine_scheduler import CosineNoiseScheduler, SchedulerConfig
import torch 


config = NoiseSchedulerConfig(
    num_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    zero_snr= True
)

config_cos = SchedulerConfig(
    num_timesteps=1000,
    cosine_s = 2e-8,
    zero_snr= True
)


config_no = NoiseSchedulerConfig(
    num_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    zero_snr= False
)

config_cos_no = SchedulerConfig(
    num_timesteps=1000,
    cosine_s = 2e-8,
    zero_snr= False
)



linear_scheduler = LinearNoiseScheduler(config)
cos_scheduler = CosineNoiseScheduler(config_cos)
linear_scheduler_no = LinearNoiseScheduler(config_no)
cos_scheduler_no = CosineNoiseScheduler(config_cos_no)

im = torch.zeros(10,1,28,28)
noise = torch.randn_like(im)

t = linear_scheduler.sample_timestep(im)
im =  linear_scheduler.add_noise(im,noise,t)



# |%%--%%| <ckKUAACuHC|TxOjZd8xKb>

import matplotlib.pyplot as plt

plt.plot(linear_scheduler.sqrt_alpha_cum_prod.squeeze())
plt.plot(cos_scheduler.sqrt_alpha_cum_prod.squeeze())
plt.plot(linear_scheduler_no.sqrt_alpha_cum_prod.squeeze())
plt.plot(cos_scheduler_no.sqrt_alpha_cum_prod.squeeze())





#|%%--%%| <TxOjZd8xKb|bA3hcBiBTU>

print(linear_scheduler.sqrt_alpha_cum_prod.min())
print(cos_scheduler.sqrt_alpha_cum_prod.min())
print(linear_scheduler_no.sqrt_alpha_cum_prod.min())
print(cos_scheduler_no.sqrt_alpha_cum_prod.min())

print(linear_scheduler.sqrt_alpha_cum_prod.max())
print(cos_scheduler.sqrt_alpha_cum_prod.max())
print(linear_scheduler_no.sqrt_alpha_cum_prod.max())
print(cos_scheduler_no.sqrt_alpha_cum_prod.max())

# |%%--%%| <bA3hcBiBTU|nBGaYarOf7>
