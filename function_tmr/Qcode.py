import torch

power = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def float2Qcode(input, Q, data_width):
    # assert data_width % 8 == 0, "data width must be integral multiple of 8"
    # print(type(Q))
    #p = power[Q]
    p = 2**Q
    results = round(input*p)

    boundary = 2**(data_width-1)
    results = clamp(results, -boundary, boundary-1)
    return results

def clamp(a, min_, max_):
    if a < min_:
        a = min_
    if a > max_:
        a = max_
    return a

def Qcode2float(input, Q, data_width):
    # assert data_width%8 == 0, "data_width must be integral multiple of 8"
    # assert 0<q_encode<data_width, 'q_encode need be (0, data_width)'
    #p = power[Q]
    p = 2**Q
    results = input / p
    return results


def Qdetermine(input):
    Pabs = input.abs()
    Pmax = Pabs.max()
    if 1/pow(2, -10) >= Pmax:
        return -3
    if 1/pow(2, 10) < Pmax:
        return 17
    for i in range(-10, 10):
        if 1/(2**i) < Pmax <= 1/(2**(i-1)):
            return int(i+7)
    
