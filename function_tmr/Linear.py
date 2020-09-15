import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .fc import linear
import torch.nn.functional as F
from .errorInsert import insertError_fc as insertError
from .errorInsert import f2Q,Q2f

class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, data_width=8, flagErr=3, probs=1e-7):
        weight_new = self.weight.clone()
        #weight_new = insertError(weight_new, probs, data_width)
        input_new = input.clone()
        bias_new = self.bias.clone()
        #input_new = insertError(input_new, probs, data_width)

        weight_new = f2Q(weight_new,data_width)
        input_new = f2Q(input_new,data_width)
        bias_new = f2Q(bias_new,15)
        #y_correct = F.linear(input, self.weight, self.bias)
        
        #y_noise = F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        y_noise = F.linear(input_new, weight_new, bias_new)
        y_noise = Q2f(y_noise,data_width)
        #y_noise = insertError(y_noise, probs, 16)

        ''' 
        weight_new2 = insertError(weight_new, probs, data_width)
        input_new2 = insertError(input_new, probs, data_width)
        y_noise2 = F.linear(input_new2, weight_new2, self.bias)
        y_noise2 = insertError(y_noise2, probs, 16)        

        weight_new3 = insertError(weight_new, probs, data_width)
        input_new3 = insertError(input_new, probs, data_width)
        
        y_noise3 = F.linear(input_new3, weight_new3, self.bias)
        y_noise3 = insertError(y_noise3, probs, 16)

        raws, cols = y_noise1.size()
        y_noise = y_noise1.clone()
        for i in range(raws):
            for j in range(cols):
                if (y_noise1[i][j] != y_noise2[i][j] and y_noise1[i][j] != y_noise3[i][j] and y_noise2[i][j] != y_noise3[i][j]):
                    print('...',i,j,y_noise1[i][j],y_noise2[i][j].item(),y_noise3[i][j].item())
                    y_noise[i][j] = (y_noise1[i][j] + y_noise2[i][j] + y_noise3[i][j]) / 3
                else:
                    if (y_noise1[i][j] == y_noise2[i][j]):
                        y_noise[i][j] = y_noise1[i][j]
                    elif (y_noise1[i][j] == y_noise3[i][j]):
                        y_noise[i][j] = y_noise1[i][j]
                    elif (y_noise2[i][j] == y_noise3[i][j]):
                        y_noise[i][j] = y_noise2[i][j]
   
        '''
        #y = y_noise.detach() + y_correct - y_correct.detach()
        return y_noise

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
