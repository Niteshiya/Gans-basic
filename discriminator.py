import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter



class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,16),
            nn.LeakyReLU(0.1),
            nn.Linear(16,1),
            nn.Sigmoid()
            )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, z_dim,img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,512),
            nn.LeakyReLU(0.1),
            nn.Linear(512,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),
            nn.Tanh()
            )
    def forward(self,x):
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "gpu"

print(f"Device selected : {device}")

lr=3e-4
z_dim=64
image_dim= 28*28*1
batch_size=32
num_epoch = 80

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim,image_dim).to(device)

fixed_noise = torch.randn((batch_size,z_dim)).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]
    )

dataset = datasets.MNIST(root="dataset/",transform=transforms,download=True)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
opt_disc = optim.Adam(disc.parameters(),lr=lr)
opt_gen = optim.Adam(gen.parameters(),lr=lr)

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MINST/fake")


writer_real = SummaryWriter(f"runs/GAN_MINST/real")

step=0

print(loader)
for epoch in range(num_epoch):
    for batch_idx ,(real,_) in enumerate(loader):
        real = real.view(-1,784).to(device)
        batch_size = real.shape[0]

        #train generator ---> max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn(batch_size,z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake,torch.zeros_like(disc_fake))

        lossD = (lossD_real+lossD_fake)/2

        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        #Train generator : min log(1-D(G(z))) <--> max log(D(G(z)))

        output = disc(fake).view(-1)
        lossG = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epoch}]==||||||== LossD: {lossD:4.3f},lossG: {lossG:4.3f}")
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28)
                data = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)
                img_grid_real = torchvision.utils.make_grid(data,normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images",img_grid_fake,global_step=step
                    )
                writer_real.add_image(
                    "Mnist Real Images",img_grid_real,global_step=step
                    )
                step+=1





























