# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from function_copy import conv
import torch.nn.functional as F
from .errorInsert import insertError,f2Q,Q2f
from collections import Counter
import numpy as np
import datetime
class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
    '''
    def TMR(self,input1,input2,input3):
        t = datetime.datetime.now()
        input1 = input1.cpu()
        input2 = input2.cpu()
        input3 = input3.cpu()
        b,c,w,h = input1.size()
        input1 = input1.view(b*c, -1)
        input2 = input2.view(b*c, -1)
        input3 = input3.view(b*c, -1)
        raws, cols = input1.size()
        input_copy = input1.clone()
        print(input_copy.type())
        input_cat = torch.stack((input1, input2, input3), dim=0)
        t1 = datetime.datetime.now()
        print('process',t1-t)
        for i in range(raws):
            for j in range(cols):
                input_list = []
                com_tensor = input_cat[:, i, :][:, j]
                com_np = com_tensor.detach().numpy()
                input_list = com_np.tolist()
                b = dict(Counter(input_list))
                # 只展示重复元素
                repeat_list = [key for key, value in b.items() if value > 1]
                if (repeat_list == []):
                    input_copy[i][j] = round(sum(input_list)/ 3)
                else:
                    input_copy[i][j] = repeat_list[0]
                    #print(i,j,input_list,repeat_list)
        print(datetime.datetime.now()-t1)
        print(input_copy.type())
        input_copy = torch.reshape(input_copy,(b,c,w,h))
        input_copy = input_copy.cuda()
        return input_copy
    '''

    def TMR(self,input1,input2,input3,probs):
        input1 = insertError(input1, probs, 16)
        input2 = insertError(input2, probs, 16)
        input3 = insertError(input3, probs, 16)
        b, c, raws, cols = input1.size()
        input_copy = input1.clone()
        t1 = datetime.datetime.now()
        for x in range(b):
            for y in range(c):
                for i in range(raws):
                    for j in range(cols):
                        if (input1[x][y][i][j] != input2[x][y][i][j] and input1[x][y][i][j] != input3[x][y][i][j] and input2[x][y][i][j] != input3[x][y][i][j]):
                            #print('...',i,j,input1[x][y][i][j],input2[x][y][i][j].item(),input3[x][y][i][j].item())
                            input_copy[x][y][i][j] = (input1[x][y][i][j] + input2[x][y][i][j] + input3[x][y][i][j]) / 3
                        else:
                            if (input1[x][y][i][j] == input2[x][y][i][j]):
                                input_copy[x][y][i][j] = input1[x][y][i][j]
                            elif (input1[x][y][i][j] == input3[x][y][i][j]):
                                input_copy[x][y][i][j] = input1[x][y][i][j]
                            elif (input2[x][y][i][j] == input3[x][y][i][j]):
                                input_copy[x][y][i][j] = input2[x][y][i][j]
        #print(datetime.datetime.now()-t1)
        return input_copy

    def new_conv(self,weight_new,input_new,probs,data_width):
        weight_new1 = insertError(weight_new, probs, data_width)
         
        input_new1 = insertError(input_new, probs, data_width)
        
        y_noise1 = F.conv2d(input_new1, weight_new1, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return y_noise1

    def forward(self, input, data_width=8, flagErr=3, probs=1e-7):
        weight_new = self.weight.clone()
        input_new = input.clone()
        bias_new = self.bias.clone()

        weight_new = f2Q(weight_new,data_width)
        input_new = f2Q(input_new,data_width)
        bias_new = f2Q(bias_new,15)

        y_correct = F.conv2d(input_new, weight_new, bias_new, self.stride,self.padding, self.dilation, self.groups)  
        y_correct = Q2f(y_correct,data_width) 
        #y_correct1 = F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        '''
        y_noise1 = self.new_conv(weight_new,input_new,probs,data_width)
         
        y_noise2 = self.new_conv(weight_new,input_new,probs,data_width)
        y_noise3 = self.new_conv(weight_new,input_new,probs,data_width)
        
        y_noise = self.TMR(y_noise1,y_noise2,y_noise3,probs)
        '''
        #y = y_noise.detach() + y_correct - y_correct.detach()
        return y_correct

