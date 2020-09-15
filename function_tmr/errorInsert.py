import numpy as np
from .Qcode import float2Qcode, Qcode2float, Qdetermine
import torch
power = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def insertError(input, probs, data_width=8):
    
    b,c,raws, cols = input.size()
    Q = data_width-1
    input_copy = input.clone()
    for x in range(b):
        for y in range(c):
            for i in range(raws):
                rawErrorList = randomGenerater(cols, probs)
                if rawErrorList:
                    for (j, errorBit) in rawErrorList:
                        #print('xx',input_copy[i, j].item())
                        input_copy[x][y][i][j] = insert_fault(input_copy[x][y][i][j].item(), errorBit, Q, data_width)
                        #print('xx',input_copy[i, j].item())
                else:
                    pass
    return input_copy

def insertError_fc(input, probs, data_width=8):

    raws, cols = input.size()
    Q = data_width-1
    input_copy = input.clone()
    for i in range(raws):
        rawErrorList = randomGenerater(cols, probs)
        if rawErrorList:
            for (j, errorBit) in rawErrorList:
                #print('xx',input_copy[i, j].item())
                input_copy[i][j] = insert_fault(input_copy[i][j].item(), errorBit, Q, data_width)
                #print('xx',input_copy[i, j].item())
        else:
            pass

    return input_copy


def randomGenerater(size, probs, data_width = 8):
    errorlist = []
    for i in range(size):
        if(np.random.rand() < probs):
            errorlist.append((i, int(np.random.rand() * data_width)))
    return tuple(errorlist)

def insert_fault(data, errorbit, Q, data_width):
    data = float2Qcode(data, Q, data_width)
    sign_bit = data_width-1
    if errorbit == sign_bit:
        bitmask = -(2**errorbit)
    else:
        bitmask = 2**errorbit
    value = int(data) ^ int(bitmask)
    return Qcode2float(value, Q, data_width)
  
def f2Q(input, data_width, qcode=None):
    import math
    if qcode is None:
        qcode = data_width - torch.ceil(math.log(input.abs().max()) + 1 - 1e-5)
    mul_n = 2**Q  #矩阵要乘的系数，（-mul_n,mul_n-1)
    min_n = -mul_n
    max_n = mul_n -1
    Qp = 2 ** (data_width - 1) - 1
    Qn = -(2 ** (data_widht - 1))
    input = (input * (2 ** qcode)).round().clamp(Qn, Qp) #Q编码并四舍五入
    return input, qcode

def Q2f(input, qcode_i, qcode_w):
    input = input * 2 ** (-qcode_i - qcode_w)
    return input

'''
def insert_fault(data, errorbit, Q, data_width):
    data = float2Qcode(data, Q, data_width)
    sign_bit = data_width-1
    if errorbit == sign_bit:
        bitmask = -128
    else:
        bitmask = int(power[errorbit])
    value = int(data) ^ bitmask
    return Qcode2float(value, Q, data_width)
'''
