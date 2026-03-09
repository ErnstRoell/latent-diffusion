from datasets.pcmnist import DataConfig, create_dataset
import matplotlib.pyplot as plt

config = DataConfig(
    root="./data",
    raw="./data/raw",
    num_pts=128,
    module="datasets.mnist",
    batch_size=64,
)

train_pc, _ = create_dataset(config, dev=True)


# create_dataset(config, dev=False)


# print(72 * "=")
# print("Data Configuration")
# print_config(config)
# print(72 * "=")

# get_dataloaders(config, dev=False)
# get_dataloaders(config, dev=True)
# |%%--%%| <96gCkYrZWC|QucJ5E3xG2>


plt.scatter(train_pc[1, :, 0], train_pc[1, :, 1])
