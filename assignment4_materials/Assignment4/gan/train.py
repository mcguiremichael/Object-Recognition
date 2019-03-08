import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.misc
import numpy as np
import torch
import torch.optim as optim

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Train loop for GAN.
    
    The loop will consist of two steps: a discriminator step and a generator step.
    
    (1) In the discriminator step, you should zero gradients in the discriminator 
    and sample noise to generate a fake data batch using the generator. Calculate 
    the discriminator output for real and fake data, and use the output to compute
    discriminator loss. Call backward() on the loss output and take an optimizer
    step for the discriminator.
    
    (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.
    
    You will need to reshape the fake image tensor outputted by the generator to 
    be dimensions (batch_size x input_channels x img_size x img_size).
    
    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations.
    
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    lambda1 = lambda epoch: max(0.95 ** epoch, 0.25)
    scheduler_G = optim.lr_scheduler.LambdaLR(G_solver, lr_lambda=lambda1)
    scheduler_D = optim.lr_scheduler.LambdaLR(D_solver, lr_lambda=lambda1)
    iter_count = 0
    reference_noise = sample_noise(1, noise_size).to(device)
    save_ref_every = 5
    for epoch in range(num_epochs):
        scheduler_G.step()
        scheduler_D.step()
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device)  # normalize
            
            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images = None
            
            ####################################
            #          YOUR CODE HERE          #
            ####################################
            
            ### Discriminator
            D_solver.zero_grad()
            noise = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(noise).reshape((batch_size, input_channels, img_size, img_size))

            
            logits_real = D(real_images)
            logits_fake = D(fake_images)
            d_error = discriminator_loss(logits_real, logits_fake)
            d_error.backward()
            D_solver.step()
            
            ### Generator
            """
            (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.
            """
            G_solver.zero_grad()
            noise = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(noise).reshape((batch_size, input_channels, img_size, img_size))
            logits_fake = D(fake_images)
            g_error = generator_loss(logits_fake)
            g_error.backward()
            G_solver.step()
            
            ##########       END      ##########
            
            # Logging and output visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
                
            if (iter_count % save_ref_every == 0):
                image = G(reference_noise)
                img_array = image.cpu().data.numpy()[0,:,:,:]
                img_array = np.swapaxes(np.swapaxes(img_array, 0, 1), 1, 2)
                scipy.misc.imsave('outfile' + str(int(iter_count / save_ref_every)) + '.png', img_array)
                
            iter_count += 1
        torch.save(D.state_dict(), "mydiscriminator.pt")
        torch.save(G.state_dict(), "mygenerator.pt")
