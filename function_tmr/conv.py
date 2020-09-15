import numpy as np
import torch
import torch.nn.functional as F
from .errorInsert import insertError

def conv2d(input_, weight_, bias, stride, padding, dilation, groups, probs, data_width, flagErr):
    b, c_in, h_in, w_in = input_.data.size()
    c_out, C_, h, w = weight_.data.size()
    print("function_copy_conv")
    h_out = np.int(np.floor((h_in + 2 * padding[0] - dilation[0] * (h - 1) - 1) / stride[0]) + 1)
    w_out = np.int(np.floor((w_in + 2 * padding[1] - dilation[1] * (w - 1) - 1) / stride[1]) + 1)

    input_ = F.pad(input_, (padding[1], padding[1], padding[0], padding[0]), 'constant', 0)

    weight = weight_.view(c_out, -1).clone()
    imagePatch = torch.autograd.Variable(torch.zeros(b, h_out * w_out, c_in * h * w))
    convolvedimage = torch.autograd.Variable(torch.zeros(groups, c_out//groups, b * h_out * w_out))
    imagePatch = imagePatch.cuda()
    convolvedimage = convolvedimage.cuda()
    for im in range(b):
        for row in range(h_out):
            row1 = row * stride[0]
            for col in range(w_out):
                col1 = col * stride[1]
                # the im, row, col: help to choose a convolved piece from input
                imagePatch[im, row * w_out + col] = input_[im, :, row1:row1 + h, col1:col1 + w].contiguous().view(-1)
    
    imagePatch = imagePatch.contiguous().view(b * h_out * w_out * groups, (c_in//groups) * h * w)
    
    if(flagErr == 2 or flagErr == 3):
        weight = insertError(weight, probs, data_width)
    if(flagErr == 1 or flagErr == 3):
        imagePatch = insertError(imagePatch, probs, data_width)
    
    if groups > 1:
        weight = weight.view(groups, c_out//groups, -1)
        for i in range(groups):
            imagePatch = imagePatch.contiguous().view(b * h_out * w_out, groups, (c_in//groups) * h * w)
            convolvedimage[i, :, :] = imagePatch[:, i, :].matmul(weight[i, :, :].t()).t()
        convolvedimage = convolvedimage.contiguous().view(c_out, b * h_out * w_out)
    else:
        convolvedimage = imagePatch.matmul(weight.t()).t()  # (c_out, b*h_out*w_out)

    if bias is not None:
        convolvedimage = convolvedimage + bias.contiguous().view(-1, 1) #(c_out//groups, b*h_out*w_out*groups)
    '''
    if(flagErr == 1 or flagErr == 3):
        convolvedimage = insertError(convolvedimage, probs, data_width=16)
    '''
    convolvedimage = torch.transpose(convolvedimage.contiguous().view(c_out, b, h_out, w_out), 1, 0) #(b, c_out, h_out, w_out)

    return convolvedimage
