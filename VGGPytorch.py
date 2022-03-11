import torch
import torch.nn as nn
import config

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

model_weight_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

class VGGNet(nn.Module):
    def __init__(self, in_channels=3, VGGtype="VGG19", init_weights=None, batch_norm=False, num_classes=1000, feature_mode=False):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm

        self.features = self.create_conv_layers(VGG_types[VGGtype])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def forward(self, x):
        if not self.feature_mode:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        elif self.feature_mode == True and self.batch_norm == False:
            module_list = list(self.features.modules())
            #print(module_list[1:27])
            for layer in module_list[1:27]: # conv4_4 Feature maps
                x = layer(x) 
        else:
            raise ValueError('Feature mode does not work with batch norm enabled. Set batch_norm=False and try again.')

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        batch_norm = self.batch_norm

        for x in architecture:
            if type(x) == int: # Number of features
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ]

                if batch_norm == True:
                    # Back at that time Batch Norm was not invented
                    layers += [nn.BatchNorm2d(x),nn.ReLU(),]
                else:
                    layers += [nn.ReLU()]

                in_channels = x #update in_channel

            elif x == "M": # Maxpooling
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = VGGNet(in_channels=3, VGGtype="VGG19", num_classes=1000, batch_norm=False, feature_mode=True).to(device)
    model = VGGNet(in_channels=3, VGGtype="VGG19", init_weights=config.VGG_WEIGHTS, batch_norm=False, feature_mode=True).to(device)
    #print(model)
    x = torch.randn(2, 3, 224, 224).to(device)
    print(model(x).shape)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())