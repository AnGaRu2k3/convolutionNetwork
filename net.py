import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 224 * 224, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, net_configs, mlp_configs, mlp_dropout_rate, conv_dropout_rate, max_pool_stride):
        super(CNN, self).__init__()
        conv_layers = []
        current_in_channels = in_channels
        for layer_config in net_configs['conv_layers']:
            conv_layers.append(nn.Conv2d(current_in_channels, layer_config['out_channels'], kernel_size=layer_config['kernel_size'], stride=layer_config['stride'], padding=layer_config['padding']))
            conv_layers.append(nn.BatchNorm2d(layer_config['out_channels']))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=max_pool_stride))
            conv_layers.append(nn.Dropout(conv_dropout_rate))
            current_in_channels = layer_config['out_channels']
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Assume input image size is 224x224
        conv_output_size = 224 // (2 ** len(net_configs['conv_layers']))  # This is a simplification, assumes stride 2 for each MaxPool
        final_out_channels = net_configs['conv_layers'][-1]['out_channels']
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_out_channels * conv_output_size * conv_output_size, mlp_configs['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(mlp_dropout_rate),
            nn.Linear(mlp_configs['hidden_dim'], num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x