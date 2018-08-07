import argparse
import random
import torch
import torchvision.datasets
import torchvision.utils
import torchvision.transforms as transforms
import logging

from model import init_weights, DCGANGenerator, DCGANDiscriminator

logger = logging.getLogger(__name__)

# Set up commandline options
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--num_gen_features', type=int, default=64)
parser.add_argument('--num_dis_features', type=int, default=64)
parser.add_argument('--z_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--clip', type=float, default=0.01)
parser.add_argument('--leak', type=float, default=0.2)
parser.add_argument('--num_critic', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gen_model', default='')
parser.add_argument('--dis_model', default='')
parser.add_argument('--data_dir', default='')
parser.add_argument('--out_dir', default='wgan')
parser.add_argument('--seed', type=int)
parser.add_argument('--sample_every', type=int, default=500)
parser.add_argument('--save_every', type=int, default=100)

opt = parser.parse_args()
logger.error(opt)

cuda = opt.cuda and torch.cuda.is_available()
size = int(opt.image_size)
batch_size = int(opt.batch_size)
num_epoch = int(opt.num_epoch)
num_gen_features = int(opt.num_gen_features)
num_dis_features = int(opt.num_dis_features)
z_size = int(opt.z_size)
learning_rate = float(opt.lr)
clip_val = float(opt.clip)
leak_val = float(opt.leak)
num_critic = int(opt.num_critic)
num_workers = int(opt.num_workers)
num_channels = 3
sample_every = int(opt.sample_every)
save_every = int(opt.save_every)

if opt.seed is None:
    seed = random.randint(1, 100000)
else:
    seed = opt.seed

random.seed(seed)
torch.manual_seed(seed)

logger.error('Seed: %d' % seed)

device = torch.device("cuda:0" if cuda else "cpu")

if __name__ == '__main__':
    # Load dataset
    dataset = torchvision.datasets.ImageFolder(root=opt.data_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cuda,
        )


    # Initialize models
    generator = DCGANGenerator(z_size, num_gen_features, num_channels).to(device)
    discriminator = DCGANDiscriminator(num_dis_features, num_channels, leak_val).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    if opt.gen_model != '':
        generator.load_state_dict(torch.load(opt.gen_model))
    if opt.dis_model != '':
        discriminator.load_state_dict(torch.load(opt.gen_model))

    # Prep optimizer
    optim_gen = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
    optim_dis = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)
    
    
    # Prep tensors
    one = torch.FloatTensor(1).to(device)
    mone = one * -1
    fixed_noise = torch.randn(batch_size, z_size, 1, 1).to(device)
    
    
    gen_itr = 0
    for epoch in range(num_epoch):
        i = 0
        data_itr = iter(dataloader)
        while i < len(dataloader):
            # Unfreeze discriminator
            for param in discriminator.parameters():
                param.requires_grad_(True)
            # Update discriminator
            j = 0
            while j < num_critic and i < len(dataloader):
                j += 1
                # Reset gradients
                discriminator.zero_grad()

                # Train discriminator with real data
                batch = data_itr.next()
                i += 1
                real_batch, _ = batch # 2nd element is a label which we don't use

                real_batch = real_batch.to(device)
                input_real = real_batch.new_tensor(real_batch, requires_grad=True)

                error_dis_real = discriminator(input_real)
                error_dis_real = error_dis_real.mean()
                
                # Train discriminator with ouput of generator
                noise = torch.randn(batch_size, z_size, 1, 1).to(device)
                fake_batch = generator(noise.detach())
                input_fake = fake_batch.new_tensor(fake_batch, requires_grad=True)
                
                error_dis_fake = discriminator(input_fake)
                error_dis_fake = error_dis_fake.mean()

                error_dis = -(error_dis_real - error_dis_fake)
                error_dis.backward()

                optim_dis.step()
                
                # Clip weights (not gradients...seems weird)
                for param in discriminator.parameters():
                    param.data.clamp_(-clip_val, clip_val)
                
                #logger.error('test Epoch:%d/%d Batch:%d/%d Loss_D:%.4f Loss_D_real:%.4f Loss_D_fake:%.4f' % (epoch, num_epoch, i, len(dataloader), error_dis.item(), error_dis_real.item(), error_dis_fake.item()))
                

            # Update generator
            # Freeze discriminator
            for param in discriminator.parameters():
                param.requires_grad_(False)

            # Reset gradients
            generator.zero_grad()
            # Train generator
            input_noise = torch.randn(batch_size, z_size, 1, 1, requires_grad=True).to(device)
            gen_out = generator(input_noise)
            error_gen = discriminator(gen_out)
            error_gen = error_gen.mean()
            error_gen.backward()
            gen_itr += 1

            logger.error('Epoch:%d/%d Batch:%d/%d Loss_D:%.4f Loss_G:%.4f Loss_D_real:%.4f Loss_D_fake:%.4f' % (epoch, num_epoch, i, len(dataloader), error_dis.item(), error_gen.item(), error_dis_real.item(), error_dis_fake.item()))

            if gen_itr % sample_every == 0:
                # real_sample = real_batch
                # torchvision.utils.save_image(real_sample,
                #         '%s/sample/real_epoch_%d_genitr_%d.png' % (opt.out_dir, epoch, gen_itr),
                #         normalize=True)
                fake_sample = generator(fixed_noise)
                torchvision.utils.save_image(fake_sample.detach(),
                        '%s/sample/fake_epoch_%d_genitr_%d.png' % (opt.out_dir, epoch, gen_itr),
                        normalize=True)


        # Check point
        if epoch % save_every == 0: 
            torch.save(generator.state_dict(), '%s/gen_epoch_%d.model' % (opt.out_dir, epoch))
            torch.save(discriminator.state_dict(), '%s/dis_epoch_%d.model' % (opt.out_dir, epoch))


