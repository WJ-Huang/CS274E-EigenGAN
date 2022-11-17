# %%
import torch
import torch.nn.functional as F

# %%
def calculateHingeLoss():

    def calculateDLoss(real_pred, fake_pred):
        return F.relu(1 - real_pred).mean(), F.relu(1 + fake_pred).mean()

    def calculateGLoss(fake_pred):
        return F.relu(1 - fake_pred).mean()

    return calculateDLoss, calculateGLoss

# %%
def calculateR1Loss(output, input):
    grad, = torch.autograd.grad(outputs=output.sum(), inputs=input, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()