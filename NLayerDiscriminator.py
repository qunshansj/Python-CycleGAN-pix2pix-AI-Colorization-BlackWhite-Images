
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:  ##判断归一化层类别，如果是BN则不需要bias

            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:

            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4 ##卷积核大小

        padw = 1 ##填充大小

        ## 第一个卷积层

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1

        nf_mult_prev = 1

        ## 中间2个卷积层

        for n in range(1, n_layers):  ##逐渐增加通道宽度，每次扩充为原来两倍

            nf_mult_prev = nf_mult

            nf_mult = min(2 ** n, 8)

            sequence += [

                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),

                norm_layer(ndf * nf_mult),

                nn.LeakyReLU(0.2, True)

            ]

        nf_mult_prev = nf_mult

        nf_mult = min(2 ** n_layers, 8)

        ## 第五个卷积层

        sequence += [

            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),

            norm_layer(ndf * nf_mult),

            nn.LeakyReLU(0.2, True)

        ]

        ## 输出单通道预测结果图

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] 

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return self.model(input)
