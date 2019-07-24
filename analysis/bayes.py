import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NoiseLoss(nn.Module):
  # need the scale for noise standard deviation
  # scale = noise  std
    def __init__(self, params, noise_std, observed=None):
        super(NoiseLoss, self).__init__()
        self.observed = observed
        self.noise_std = noise_std

    def forward(self, params,  observed=None):
    # scale should be sqrt(2*alpha/eta)
    # where eta is the learning rate and alpha is the strength of drag term
        if observed is None:
            observed = self.observed

#         assert scale is not None, "Please provide scale"
        noise_loss = 0.0
        for var in params:
            # This is scale * z^T*v
            # The derivative wrt v will become scale*z
#             _noise = noise.normal_(0.,self.noise_std)
            _noise = self.noise_std*torch.randn(1)
            noise_loss += torch.sum(Variable(_noise)*var)
        noise_loss /= observed
        return noise_loss

class PriorLoss(nn.Module):
  # negative log Gaussian prior
    def __init__(self, prior_std=1., observed=None):
        super(PriorLoss, self).__init__()
        self.observed = observed
        self.prior_std = prior_std

    def forward(self, params, observed=None):
        if observed is None:
            observed = self.observed
        prior_loss = 0.0
        for var in params:
            prior_loss += torch.sum(var*var/(self.prior_std*self.prior_std))
        prior_loss /= observed
        return prior_loss