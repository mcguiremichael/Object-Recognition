import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    targets_real = torch.ones(logits_real.shape).to(logits_real.device)
    targets_fake = torch.zeros(logits_fake.shape).to(logits_real.device)
    loss = (bce_loss(logits_real, targets_real) + bce_loss(logits_fake, targets_fake))
    return loss
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    targets = torch.ones(logits_fake.shape).to(logits_fake.device)
    loss = bce_loss(logits_fake, targets)
    return loss
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    #return - bce_loss(logits_fake,
    ##########       END      ##########
    


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    targets_real = torch.ones(scores_real.shape).to(scores_real.device)
    targets_fake = torch.zeros(scores_fake.shape).to(scores_fake.device)
    
    loss = 0.5 * ( torch.mean((scores_real - targets_real) ** 2) + torch.mean((scores_fake - targets_fake) ** 2) )
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    targets_fake = torch.ones(scores_fake.shape).to(scores_fake.device)
    loss = 0.5 * torch.mean( (scores_fake - targets_fake) ** 2 )
    ##########       END      ##########
    
    return loss
