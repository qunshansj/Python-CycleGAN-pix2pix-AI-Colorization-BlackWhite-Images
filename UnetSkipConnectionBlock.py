
UnetSkipConnectionBlock是跳层连接的模块，它的定义如下：

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,

                 submodule=None,outermost=False,innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost

        if type(norm_layer) == functools.partial:

            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:

            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:

            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,

                             stride=2, padding=1, bias=use_bias)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)

        upnorm = norm_layer(outer_nc)

        if outermost:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,

                                        kernel_size=4, stride=2,

                                        padding=1)

            down = [downconv]

            up = [uprelu, upconv, nn.Tanh()]

            model = down + [submodule] + up

        elif innermost:

            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,

                                        kernel_size=4, stride=2,

                                        padding=1, bias=use_bias)

            down = [downrelu, downconv]

            up = [uprelu, upconv, upnorm]

            model = down + up

        else:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,

                                        kernel_size=4, stride=2,

                                        padding=1, bias=use_bias)

            down = [downrelu, downconv, downnorm]

            up = [uprelu, upconv, upnorm]

            ##是否使用dropout

            if use_dropout:

                model = down + [submodule] + up + [nn.Dropout(0.5)]

            else:

                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):

        if self.outermost:#最外层直接输出

            return self.model(x)

        else:#添加跳层

            return torch.cat([x, self.model(x)], 1)
