class SeameseResNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(SeameseResNet, self).__init__()

        self.encoder_1 = encoder
        self.encoder_2 = encoder
        self.decoder = decoder

    def forward(self, x_1, x_2):
        e_1 = self.encoder_1(x_1)
        e_2 = self.encoder_1(x_2)
        out = self.decoder(e_1, e_2)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, last_layer: int=None):
        super(DeconvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2= nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        if last_layer is not None:
            self.conv3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=last_layer,
                                kernel_size=(1, 1),
                                stride=1, padding=0)
        else:           
            self.conv3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                stride=1,
                                padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        return self.conv3(x)


class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.con1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.con2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=1024)
        self.con3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=1024)

    def forward(self, x):
        x = F.relu(self.con1(x))
        x = self.bn1(x)
        x = F.relu(self.con1(x))
        x = self.bn2(x)
        x = F.relu(self.con2(x))
        x = self.bn2(x)
        x = F.relu(self.con3(x))
        x = self.bn3(x)

        return F.relu(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=True)
        self.deconv_block1 = DeconvBlock(1024, 512)
        self.deconv_block2 = DeconvBlock(1024, 256)
        self.deconv_block3 = DeconvBlock(512, 128)
        self.deconv_block4 = DeconvBlock(256, 128)
        self.deconv_block5 = DeconvBlock(128, 64, 1)
    
    def forward(self, x_1, x_2):
        x = torch.cat((x_1[-1], x_2[-1]), dim=1)
        x = self.deconv_block1(x)
        x = self.upsample(x)
        x = torch.cat((x, x_1[-2], x_2[-2]), dim=1)
        x = self.deconv_block2(x)
        x = self.upsample(x)
        x = torch.cat((x, x_1[-3], x_2[-3]), dim=1)
        x = self.deconv_block3(x) 
        x = self.upsample(x)
        x = torch.cat((x, x_1[0], x_2[0]), dim=1)
        x = self.upsample(x)
        x = self.deconv_block4(x)
        x = self.upsample(x)
        x = self.deconv_block5(x)
        return x
    
class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        resnet34 = models.resnet34(pretrained=True).to(device)

        #self.conv1 = resnet34.conv1
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.layer = []
        self.layer.append(resnet34.layer1)
        self.layer.append(resnet34.layer2)
        self.layer.append(resnet34.layer3)
        self.layer.append(resnet34.layer4)

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)
        for l in self.layer:
            x = l(x)
            output.append(x)
        return output
