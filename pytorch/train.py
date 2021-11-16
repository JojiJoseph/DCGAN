import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Normalize, PILToTensor
from torchvision.transforms.transforms import ConvertImageDtype

from gan import Disc, Generator
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = torchvision.datasets.MNIST("./data", download=True, transform=Compose(
    [PILToTensor(), ConvertImageDtype(torch.float), Normalize(0.5, 0.5), ]))

batch_size = 32
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=10)

if __name__ == "__main__":
    os.makedirs("./figs", exist_ok=True)
    gen = Generator().to(device)
    disc = Disc().to(device)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-4)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4)

    bce = torch.nn.BCELoss()
    for epoch in range(1_000):
        print(f"Epoch {epoch+1} ...")

        for batch_id, (real_batch, label_batch) in tqdm(enumerate(data_loader), total=len(data_loader)):

            real_batch = real_batch.to(device)
            
            # Train generator
            z = np.random.normal(size=(batch_size, 100))
            z = torch.from_numpy(z).float().to(device)
            y_g = gen(z)
            y_d = disc(y_g)
            loss = bce(y_d, torch.ones((batch_size, 1)).to(device))
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()

            # Train discriminator
            y_g = y_g.detach()
            y_d = disc(y_g)
            loss = bce(y_d, torch.zeros((batch_size, 1)).to(device))
            y_d = disc(real_batch)
            loss += bce(y_d, torch.ones((batch_size, 1)).to(device))
            opt_disc.zero_grad()
            loss.backward()
            opt_disc.step()
        torch.save(disc.state_dict(), "./disc_gan.pt")
        torch.save(gen.state_dict(), "./gen_gan.pt")

        gen.eval()
        for i in range(9):
            plt.subplot(3, 3, i+1)
            z = np.random.normal(size=(1, 100))
            z = torch.from_numpy(z).float().to(device)
            y = (gen(z)+1)/2
            y = torch.reshape(y, (28, 28, 1)).detach().cpu().numpy()
            plt.imshow(y, cmap="gray")
        plt.savefig(f"./figs/fig{epoch}.png")
        gen.train()
    plt.show()
