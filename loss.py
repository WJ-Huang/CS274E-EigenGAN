import torch
import torch.nn.functional as F

def discriminator_hinge_loss(real_pred, fake_pred):
    return F.relu(1 - real_pred).mean() + F.relu(1 + fake_pred).mean()

def generator_hinge_loss(fake_pred):
    return F.relu(1 - fake_pred).mean()

def r1_loss(output, input):
    grad = torch.autograd.grad(outputs=output.sum(), inputs=input, create_graph=True)[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()