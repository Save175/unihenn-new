from .layers import *
from .utils import *
from .constants import *

import torch
import math
import time
import sys

class HE_CNN(torch.nn.Module):
    """
    Simple CNN model with Homomorphic Encryption:
    Conv2d | AvgPool2d | Flatten | FC | Square | Approx ReLU
    """
    def __init__(self, model, Img: Cuboid, context: Context):
        super().__init__()
        self.model = model
        self.Img = Img
        self.context = context
        self.data_size = self.calculate_data_size()

    def calculate_depth(self):
        req_depth = 0
        _const = 1
        Out = self.Img.CopyNew()

        for layer_name, layer in self.model.named_children():
            layer_params = getattr(self.model, layer_name)
            
            if layer.__class__.__name__ == 'Conv2d':
                req_depth += 1
                _const = 1
                Out.z = layer_params.out_channels
                Out.h = (Out.h + 2 * layer_params.padding[0] - layer_params.kernel_size[0]) // layer_params.stride[0] + 1
                Out.w = (Out.w + 2 * layer_params.padding[1] - layer_params.kernel_size[1]) // layer_params.stride[1] + 1

            elif layer.__class__.__name__ == 'AvgPool2d':
                _const = -1
                Out.h = (Out.h + 2 * layer_params.padding - layer_params.kernel_size) // layer_params.stride + 1
                Out.w = (Out.w + 2 * layer_params.padding - layer_params.kernel_size) // layer_params.stride + 1
            
            elif layer.__class__.__name__ == 'Square':
                req_depth += 1

            elif layer.__class__.__name__ == 'ApproxReLU':
                req_depth += 2
                _const = 1

            elif layer.__class__.__name__ == 'Flatten':
                if Out.w != 1 or _const != 1:
                    req_depth += 1
                if Out.h != 1:
                    req_depth += 1

            elif layer.__class__.__name__ == 'Linear':
                req_depth += 1
            
        return req_depth

    def calculate_data_size(self):
        data_size = self.Img.size2d()

        for layer_name, layer in self.model.named_children():
            layer_params = getattr(self.model, layer_name)
            if layer.__class__.__name__ == 'AvgPool2d':
                req_size = self.Img.size2d() + (self.Img.w + 1) * (layer_params.stride - 1)
                if data_size < req_size:
                    data_size = req_size
                
            elif layer.__class__.__name__ == 'Linear':
                _size = layer.out_features * math.ceil(layer.in_features / layer.out_features)
                if data_size < _size:
                    data_size = _size
            
        return data_size
    
    def encrypt(self, plaintext):
        if isinstance(plaintext, list):
            return [self.context.encryptor.encrypt(self.context.encoder.encode(plain, self.context.scale)) for plain in plaintext]
        return self.context.encryptor.encrypt(self.context.encoder.encode(plaintext, self.context.scale))
    
    def decrypt(self, ciphertext):
        if isinstance(ciphertext, list):
            return [self.context.encoder.decode(self.context.decryptor.decrypt(cipher)).tolist() for cipher in ciphertext]
        return self.context.encoder.decode(self.context.decryptor.decrypt(ciphertext)).tolist()
    
    def forward(self, C_in: list, _time=False):
        req_depth = self.calculate_depth()
        
        if req_depth > self.context.depth:
            raise ValueError("There is not enough depth to infer the current model.")

        time_logs = {}
        start_time = time.time()
        C_out = re_depth(self.context, C_in, self.context.depth - req_depth)
        Out = Output(C_out, self.Img)
        time_logs['Drop depth'] = time.time() - start_time
        
        for layer_name, layer in self.model.named_children():
            layer_start = time.time()
            layer_params = getattr(self.model, layer_name)
            
            if layer.__class__.__name__ == 'Conv2d':
                Out = conv2d_layer_converter_(self.context, Out, self.Img, layer_params, self.data_size)
            elif layer.__class__.__name__ == 'AvgPool2d':
                Out = average_pooling_layer_converter(self.context, Out, self.Img, layer_params)
            elif layer.__class__.__name__ == 'Square':
                Out = square(self.context, Out)
            elif layer.__class__.__name__ == 'ApproxReLU':
                Out = approximated_ReLU_converter(self.context, Out)
            elif layer.__class__.__name__ == 'Flatten':
                Out = flatten(self.context, Out, self.Img, self.data_size)
            elif layer.__class__.__name__ == 'Linear':
                Out.ciphertexts[0] = fc_layer_converter(self.context, Out.ciphertexts[0], layer_params, self.data_size)

            time_logs[layer_name] = time.time() - layer_start

        total_time = time.time() - start_time
        time_logs['Total Execution'] = total_time

        return Out.ciphertexts[0], time_logs
                
    def __str__(self):
        return self.model.__str__()

