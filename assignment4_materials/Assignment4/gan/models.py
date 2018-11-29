import torch


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        FILTERS_1 = 128
        FILTERS_2 = 256
        FILTERS_3 = 512
        FILTERS_4 = 1024
        
        
        self.conv1 = torch.nn.Conv2d(input_channels,
                                        FILTERS_1,
                                        kernel_size=4,
                                        padding=1,
                                        stride=2)
                                        
        self.conv2 = torch.nn.Conv2d(FILTERS_1,
                                        FILTERS_2,
                                        kernel_size=4,
                                        padding=1,
                                        stride=2)
                                        
        self.conv2_bn = torch.nn.BatchNorm2d(FILTERS_2)
                                        
        self.conv3 = torch.nn.Conv2d(FILTERS_2,
                                        FILTERS_3,
                                        kernel_size=4,
                                        padding=1,
                                        stride=2)
                                        
        self.conv3_bn = torch.nn.BatchNorm2d(FILTERS_3)
                                        
        self.conv4 = torch.nn.Conv2d(FILTERS_3,
                                        FILTERS_4,
                                        kernel_size=4,
                                        padding=1,
                                        stride=2)
                                                       
        self.conv4_bn = torch.nn.BatchNorm2d(FILTERS_4)
                                                                     
        self.conv5 = torch.nn.Conv2d(FILTERS_4,
                                        1,
                                        kernel_size=4,
                                        padding=0,
                                        stride=1)
                  
        self.activation = torch.nn.functional.leaky_relu
                                                                       
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.activation(self.conv1(x),negative_slope=0.2)
        x = self.activation(self.conv2_bn(self.conv2(x)),negative_slope=0.2)
        x = self.activation(self.conv3_bn(self.conv3(x)),negative_slope=0.2)
        x = self.activation(self.conv4_bn(self.conv4(x)),negative_slope=0.2)
        x = self.activation(self.conv5(x),negative_slope=0.2)
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        FILTERS_1 = 1024
        FILTERS_2 = 512
        FILTERS_3 = 256
        FILTERS_4 = 128
        
        self.tconv1 = torch.nn.ConvTranspose2d(noise_dim,
                                                    FILTERS_1,
                                                    kernel_size=4,
                                                    padding=0,
                                                    stride=1)
           
        self.conv1_bn = torch.nn.BatchNorm2d(FILTERS_1)
                                                    
        self.tconv2 = torch.nn.ConvTranspose2d(FILTERS_1,
                                                    FILTERS_2,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        self.conv2_bn = torch.nn.BatchNorm2d(FILTERS_2)
        
        self.tconv3 = torch.nn.ConvTranspose2d(FILTERS_2,
                                                    FILTERS_3,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        self.conv3_bn = torch.nn.BatchNorm2d(FILTERS_3)
        
        self.tconv4 = torch.nn.ConvTranspose2d(FILTERS_3,
                                                    FILTERS_4,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        self.conv4_bn = torch.nn.BatchNorm2d(FILTERS_4)
        
        self.tconv5 = torch.nn.ConvTranspose2d(FILTERS_4,
                                                    3,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        self.activation = torch.nn.functional.leaky_relu
        
        ##########       END      ##########
    
    def forward(self, x):
        x = x.reshape(-1, self.noise_dim, 1, 1)
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.activation(self.conv1_bn(self.tconv1(x)),negative_slope=0.2)
        x = self.activation(self.conv2_bn(self.tconv2(x)),negative_slope=0.2)
        x = self.activation(self.conv3_bn(self.tconv3(x)),negative_slope=0.2)
        x = self.activation(self.conv4_bn(self.tconv4(x)),negative_slope=0.2)
        x = torch.nn.functional.tanh(self.tconv5(x))
        
        ##########       END      ##########
        
        return x
    

