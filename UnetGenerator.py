
class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):

 super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf*8,ngf*8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer

        for i in range(num_downs - 5):        

            unet_block=UnetSkipConnectionBlock(ngf*8,ngf*8,input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

            ## 逐步减小通道数，从ngf * 8到ngf

            unet_block=UnetSkipConnectionBlock(ngf*4,ngf*8,input_nc=None, submodule=unet_block, norm_layer=norm_layer)

            unet_block=UnetSkipConnectionBlock(ngf*2,ngf*4,input_nc=None, submodule=unet_block, norm_layer=norm_layer)

            unet_block=UnetSkipConnectionBlock(ngf,ngf*2,input_nc=None, submodule=unet_block, norm_layer=norm_layer)

            self.model=UnetSkipConnectionBlock(output_nc,ngf,input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer) ## 最外层

    def forward(self, input):

        """Standard forward"""

        return self.model(input)
