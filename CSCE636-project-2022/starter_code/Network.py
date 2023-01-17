### YOUR CODE HERE
# import tensorflow as tf
import math
import torch
import torch.nn.functional as F
from torch.functional import Tensor
import torch.nn as nn

"""This script defines the network.
"""
## re-structuring the names of the funciton given for my model
class MyNetwork(nn.Module):
    '''Try to implement NFNet - Normalize free resnets
    1) Change the design of bottle neck design 1 - 3- 1 to 1 - 3 - 3 - 1
    2) Have extra parameter alpha and beta for every block - alpha = 0.2 (reference from paper.) 
       beta = var(input of block l) = var(input of block l - 1) + alpha ** 2
    3) Weigh normalization 
    4) Gradient clipping

    NOTE: I think we need to do residual layer variance preservation at initialization
    https://arxiv.org/pdf/2101.08692.pdf
    https://arxiv.org/pdf/2102.06171.pdf
    Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 4n     | 4n     | 4n     | 1           |
        #filters        | 16    | 64     | 128    | 256    | num_classes |

        NENet bottle beck : 1 - 3 - 3 - 1
        choose the initialization in such way that var(f(z)) = var(z), where f is residual block and z is input to the residual block
    '''
    def __init__(self, block_size, classes, first_num_filters):
        super(MyNetwork, self).__init__()
        self.classes = classes
        self.block_size = block_size
        self.first_num_filters = first_num_filters

        self.stack_layers = nn.ModuleList()
        self.start_layer = Conv2d_modified(3, first_num_filters, 3, stride = 1, padding = 1)
        alpha = 0.2

        for i in range(3): # 3 could be varied
            ## handle the Beta here properly
            filters = 4*self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            first_stack = 1 if i == 0 else 0
            beta_start = 1 if i == 0 else 1 + self.block_size*alpha**2
            self.stack_layers.append(stack_layer(filters, strides, self.block_size, first_stack, beta = beta_start))

        self.output_layer = output_layer(filters, self.classes)
    
    def forward(self, inputs):
        
        outputs = self.start_layer(inputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

class stack_layer(nn.Module):
    """Transition occurs at the first block of """
    def __init__(self, filters, strides, block_size, first_stack=0, alpha = 0.2, beta = 1.0):
        super(stack_layer, self).__init__()
        self.block_size = block_size
        self.stack_layers = nn.ModuleList()
        projection_shortcut = None
        filter_4 = int(filters/4)
        filter_2 = int(filters/2)
        if first_stack:
            projection_shortcut = Conv2d_modified(filter_4, 
                                                 filters, 
                                                 1, 
                                                 stride=strides, 
                                                 padding = 0)

            self.stack_layers.append(NFNet_Block(filters, projection_shortcut, filter_4, strides, alpha, beta))
        else:
            projection_shortcut = Conv2d_modified(filter_2, 
                                                 filters, 
                                                 1, 
                                                 stride=strides, 
                                                 padding = 0)
            
            self.stack_layers.append(NFNet_Block(filters, projection_shortcut, filter_2, strides, alpha, beta))
        
        beta = 1 + alpha**2
        for i in range(1, block_size):
            self.stack_layers.append(NFNet_Block(filters, None, filters, 1, alpha, beta))
            beta = beta + alpha**2

    
    def forward(self, inputs):
        outputs = self.stack_layers[0](inputs)
        for key in range(1, self.block_size):
            outputs = self.stack_layers[key](outputs)
        return outputs

class NFNet_Block(nn.Module):
    def __init__(self, filters, projection_shortcut, in_channel, strides, alpha, beta):
        super(NFNet_Block, self).__init__()
        
        self.projection_shortcut = projection_shortcut
        self.beta = beta
        self.alpha = alpha
        filter_4 = int(filters/4)
        self.block_seq = nn.Sequential(
            nn.ReLU(),
            Conv2d_modified(in_channel, filter_4, 1, stride = strides),
            nn.ReLU(),
            Conv2d_modified(filter_4, filter_4, 3, stride=1, padding = 1),
            nn.ReLU(),
            Conv2d_modified(filter_4, filter_4, 3, stride=1, padding = 1),
            nn.ReLU(),
            Conv2d_modified(filter_4, filters, 1, stride=1)
        )


    def forward(self, inputs):

        outputs = inputs/self.beta

        outputs = self.block_seq(outputs)
        outputs = self.alpha*outputs

        if self.projection_shortcut is not None:
            inputs = self.projection_shortcut(inputs/self.beta)
        outputs = outputs + inputs

        return outputs

class Conv2d_modified(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation = 1, groups= 1, bias = True, gain = 1.712858):
        super(Conv2d_modified, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        #print("these are weights of the Conv2d {}".format(self.weight))
        nn.init.kaiming_normal_(self.weight, nonlinearity = 'relu')
        #print("these are weights after the Conv2d {}".format(self.weight))
        self.gain = gain
        k = kernal_size[0] * kernal_size[1] if type(kernel_size) == tuple else kernel_size**2
        self.N = math.sqrt(k * in_channels)
        #self.N = torch.sqrt(self.N)
    def forward(self, inputs):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5

        weight = (weight / std.expand_as(weight))/self.N

        weight = self.gain*weight
        return F.conv2d(inputs, weight ,self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class output_layer(nn.Module):
    
    def __init__(self, filters, num_classes) -> None:
        super(output_layer, self).__init__()
        
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(filters, num_classes)
        
    
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.avg_pool(inputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.fc(outputs)
        return outputs
      

### END CODE HERE