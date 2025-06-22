import torch
import torch.nn as nn
from swin import SwinTransformer
import torch.nn.functional as F
from MIRNet import *


class UNet_emb(nn.Module):


    def __init__(self, in_channels=3, out_channels=3,bias=False):


        super(UNet_emb, self).__init__()
        self.embedding_dim = 3


        self.conv1 = nn.Conv2d(256*3, 256, 3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(384, 256, 3, stride=1, padding=1)         
        self.conv1_2 = nn.Conv2d(384, 256, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(128*3, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv2_2= nn.Conv2d(192, 128, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64*3, 64, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(96, 64, 3, stride=1, padding=1) 
        self.conv3_2 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(48, 24, 3, stride=1, padding=1)


        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(96, 48, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(48, 24, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(24, 12, 3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(12, 12, 3, stride=1, padding=1)

        self.in_chans = 3
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ReLU=nn.ReLU(inplace=True)
        #IN 实例标准化操作,特征调制模块,对CNN特征进行的操作
        self.IN_1=nn.InstanceNorm2d(48, affine=False)
        self.IN_2=nn.InstanceNorm2d(96, affine=False)
        self.IN_3=nn.InstanceNorm2d(192, affine=False)


        self.PPM1 = PPM(32, 8, bins=(1,2,3,4))
        self.PPM2 = PPM(64, 16, bins=(1,2,3,4))
        self.PPM3 = PPM(128, 32, bins=(1,2,3,4))
        self.PPM4 = PPM(256, 64, bins=(1,2,3,4))


        self.MSRB1=MSRB(256, 3, 1, 2,bias)
        self.MSRB2=MSRB(128, 3, 1, 2,bias)
        self.MSRB3=MSRB(64, 3, 1, 2,bias)
        self.MSRB4=MSRB(32, 3, 1, 2,bias)


        self.swin_1 = SwinTransformer(pretrain_img_size=224,
                                    patch_size=2,
                                    in_chans=3,
                                    embed_dim=96,
                                    depths=[2, 2, 2],
                                    num_heads=[3, 6, 12],  #[3,6,9]?
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True, 
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0., 
                                    drop_path_rate=0.2,
                                    norm_layer=nn.LayerNorm, 
                                    ape=False,
                                    patch_norm=True,
                                    out_indices=(0, 1, 2),
                                    frozen_stages=-1,
                                    use_checkpoint=False)



        self.E_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))
        
        self.E_block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))
        
        self.E_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))



        self._block1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block5= nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
            
        self._block6= nn.Sequential(
            nn.Conv2d(46, 23, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(23, 23, 3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2))
            
        self._block7= nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))
                   

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        swin_in = x #96,192,384,768
        swin_out_1=self.swin_1(swin_in)


        # Encoder 
        swin_input_1=self.E_block1(swin_in)#32    
        swin_input_1=self.PPM1(swin_input_1)

        swin_input_2=self.E_block2(swin_input_1)
        swin_input_2=self.PPM2(swin_input_2)     

        swin_input_3=self.E_block3(swin_input_2)
        swin_input_3=self.PPM3(swin_input_3)


        swin_input_4=self.E_block4(swin_input_3)
        swin_input_4=self.PPM4(swin_input_4)

        upsample1 = self._block1(swin_input_4)


        beta_1 = self.conv1_1(swin_out_1[2])
        gamma_1 = self.conv1_2(swin_out_1[2])
        swin_input_3_refine=self.IN_3(swin_input_3)*beta_1+gamma_1





        concat3 = torch.cat((swin_input_3,swin_input_3_refine,upsample1), dim=1)


        decoder_3 = self.ReLU(self.conv1(concat3))
        upsample3 = self._block3(decoder_3)
        upsample3=self.MSRB2(upsample3)

        #2
        beta_2 = self.conv2_1(swin_out_1[1])
        gamma_2 = self.conv2_2(swin_out_1[1])
        swin_input_2_refine=self.IN_2(swin_input_2)*beta_2+gamma_2
        concat2 = torch.cat((swin_input_2,swin_input_2_refine,upsample3), dim=1)

        decoder_2 = self.ReLU(self.conv2(concat2))
        upsample4 = self._block4(decoder_2)
        upsample4=self.MSRB3(upsample4)

        #3
        beta_3 = self.conv3_1(swin_out_1[0])
        gamma_3 =self.conv3_2(swin_out_1[0]) 
        swin_input_1_refine=self.IN_1(swin_input_1)*beta_3+gamma_3
        concat1 = torch.cat((swin_input_1,swin_input_1_refine,upsample4), dim=1)


        decoder_1 = self.ReLU(self.conv3(concat1))

        upsample5 = self._block5(decoder_1)


        decoder_0 = self.ReLU(self.conv4(upsample5))

        result=self._block7(decoder_0)

        return result
           
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),

                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
          
