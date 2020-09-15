import numpy as np
import torch
import torch.nn.functional as F
from .errorInsert import insertError

def linear(input_, weight_, probs, data_width, flagErr, bias=None):
    input = input_.clone()
    weight = weight_.clone()
    raws = input.data.size()[0]
    cols = weight.data.size()[0]
    '''    
    if(flagErr == 2 or flagErr == 3):
        weight = insertError(weight, probs, data_width)
    if(flagErr == 1 or flagErr == 3):
        input = insertError(input, probs, data_width)
    '''
    linearimage = input.matmul(weight.t()) #(raws, cols)

    if bias is not None:
        linearimage = linearimage + bias.contiguous().view(1, -1)
    '''
    if(flagErr == 1 or flagErr == 3):
        linearimage = insertError(linearimage, probs, data_width)
    '''
    return linearimage
