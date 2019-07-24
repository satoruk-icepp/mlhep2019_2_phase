import numpy as np
import torch
def LabelToImages(row,col,MomentumPoint):
    images = torch.zeros(MomentumPoint.shape()[0],MomentumPoint.shape()[1],row,col)
    for image,mp in zip(images,MomentumPoint):
        for i in range(MomentumPoint.shape()[1]):
            image[i,:,:]+=mp[i].cpu()
#         print(image)
    return torch.Tensor(images)