# assert self.mid_channels[0] == self.down_channels[-1]
# assert self.mid_channels[-1] == self.down_channels[-2]
# assert len(self.down_sample) == len(self.down_channels) - 1
# assert len(self.attns) == len(self.down_channels) - 1


class BaseLightningModel(L.LightningModule):
    """Base model for VAE models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__()
        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()
        self.ect_transform = EctTransform(config=config.ectconfig, device="cuda")

        self.model = Unet(config=self.config)
        self.scheduler = LinearNoiseScheduler(config.noise_scheduler)

        self.vae = VAE.load_from_checkpoint(self.config.vae_checkpoint)

        self.visualization = []

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        return optimizer

    def forward(self, x, t):
        x = self.model(x, t)
        return x

    def general_step(self, pcs_gt, _, step: Literal["train", "test", "validation"]):
        # pcs_gt = pcs_gt[0]

        im = self.ect_transform(pcs_gt).unsqueeze(1)
        with torch.no_grad():
            im, _ = self.vae.model.encode(im)

        # Sample random noise
        noise = torch.randn_like(im)

        # Sample timestep
        t = torch.randint(
            0,
            self.config.noise_scheduler.num_timesteps,
            (im.shape[0],),
            device=self.device,
        )

        # Add noise to images according to timestep
        noisy_im = self.scheduler.add_noise(im, noise, t)
        noise_pred = self(noisy_im, t)

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss_dict = {
            f"{step}_loss": loss,
        }

        self.log_dict(
            loss_dict,
            prog_bar=True,
            batch_size=len(pcs_gt),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")
