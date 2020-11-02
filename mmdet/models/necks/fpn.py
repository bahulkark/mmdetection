import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import ConvModule, xavier_init
# from mmcv.runner import auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FPN(nn.Module):
    r"""Feature Pyramid Network.
    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.num_ins = len(in_channels)
        # self.num_outs = num_outs
        # self.relu_before_extra_convs = relu_before_extra_convs
        # self.no_norm_on_lateral = no_norm_on_lateral
        # self.fp16_enabled = False
        # self.upsample_cfg = upsample_cfg.copy()

        # if end_level == -1:
        #     self.backbone_end_level = self.num_ins
        #     assert num_outs >= self.num_ins - start_level
        # else:
        #     # if end_level < inputs, no extra level is allowed
        #     self.backbone_end_level = end_level
        #     assert end_level <= len(in_channels)
        #     assert num_outs == end_level - start_level
        # self.start_level = start_level
        # self.end_level = end_level
        # self.add_extra_convs = add_extra_convs
        # assert isinstance(add_extra_convs, (str, bool))
        # if isinstance(add_extra_convs, str):
        #     # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
        #     assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        # elif add_extra_convs:  # True
        #     if extra_convs_on_inputs:
        #         # For compatibility with previous release
        #         # TODO: deprecate `extra_convs_on_inputs`
        #         self.add_extra_convs = 'on_input'
        #     else:
        #         self.add_extra_convs = 'on_output'

        # self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()

        # for i in range(self.start_level, self.backbone_end_level):
        #     l_conv = ConvModule(
        #         in_channels[i],
        #         out_channels,
        #         1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
        #         act_cfg=act_cfg,
        #         inplace=False)
        #     fpn_conv = ConvModule(
        #         out_channels,
        #         out_channels,
        #         3,
        #         padding=1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg,
        #         inplace=False)

        #     self.lateral_convs.append(l_conv)
        #     self.fpn_convs.append(fpn_conv)

        # # add extra conv layers (e.g., RetinaNet)
        # extra_levels = num_outs - self.backbone_end_level + self.start_level
        # if self.add_extra_convs and extra_levels >= 1:
        #     for i in range(extra_levels):
        #         if i == 0 and self.add_extra_convs == 'on_input':
        #             in_channels = self.in_channels[self.backbone_end_level - 1]
        #         else:
        #             in_channels = out_channels
        #         extra_fpn_conv = ConvModule(
        #             in_channels,
        #             out_channels,
        #             3,
        #             stride=2,
        #             padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             inplace=False)
        #         self.fpn_convs.append(extra_fpn_conv)

        ## building up the first module in of the model. A_1 for input 3
        self.conv1d = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),nn.Conv2d(in_channels[3], 256, 1, stride=1, dilation = 1, padding=0)) 
        self.conv1d_1 = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),nn.Conv2d(in_channels[3], 256, 3, stride=1, dilation = 6, padding=6))
        self.conv1d_2= nn.Sequential(nn.AdaptiveAvgPool2d((8,8)), nn.Conv2d(in_channels[3], 256, 3, stride=1, dilation = 12, padding=12))
        self.conv1d_3 = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)), nn.Conv2d(in_channels[3], 256, 3, stride=1, dilation = 18, padding=18)) 
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),nn.Conv2d(in_channels[3], 256,1,stride=1,dilation = 1,padding =0),nn.BatchNorm2d(256),nn.ReLU())
        ## end first part. 

        ## C A_2
        self.relu = nn.ReLU()
        self.drop3 = nn.Dropout2d(p=0.2)
        self.conv1_1d = nn.Conv2d(1280,512, 1, stride=1,dilation  = 1,padding =0)
        self.conv2_2d = nn.Conv2d(1024,512, 3, stride=1,dilation  = 1,padding =1)
        ## C'

        ## B
        # self.drop2 = nn.Dropout2d(p=0.3)
        # self.conv1_2d = nn.Conv2d(1792,512,1 , stride=1,dilation = 1,padding =0)
        ## B'

        ## Z
        # self.bn1 = nn.BatchNorm2d(1024)
        ## Z'

        ## building up the second module in of the model. B_1 for input 0
        self.conv2_1d = nn.Sequential(nn.AdaptiveAvgPool2d((32,32)), nn.Conv2d(in_channels[0], 512, 1, stride=1,dilation = 1,padding =0))
        ## end. 
        
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    # @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # # build laterals
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]

        # # build top-down path
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #     #  it cannot co-exist with `size` in `F.interpolate`.
        #     if 'scale_factor' in self.upsample_cfg:
        #         laterals[i - 1] += F.interpolate(laterals[i],
        #                                          **self.upsample_cfg)
        #     else:
        #         prev_shape = laterals[i - 1].shape[2:]
        #         laterals[i - 1] += F.interpolate(
        #             laterals[i], size=prev_shape, **self.upsample_cfg)

        # # build outputs
        # # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]

        ## add extra... 
        
        ## perform part "A" of forward input = inputs[3] ;
        print(inputs[3].shape)
        x_1 = F.relu(self.conv1d(inputs[3]))
        x_2 = F.relu(self.conv1d_1(inputs[3]))
        x_3 = F.relu(self.conv1d_2(inputs[3]))
        x_4 = F.relu(self.conv1d_3(inputs[3]))
        x_5 = self.global_avg_pool(inputs[3])

        print(x_1.shape, x_2.shape, x_3.shape,x_4.shape, x_5.shape)

        x_6 = torch.cat((x_1, x_2, x_3, x_4, x_5), dim=-3)
        x_7 = F.relu(self.conv1_1d(x_6))
        x_8 = F.interpolate(x_7, size=32, mode='bilinear', align_corners=True)
        print(inputs[0].shape)

        x_9 = F.relu(self.conv2_1d(inputs[0]))
        print(x_9.shape)
        y = torch.cat((x_8, x_9), dim =-3)
        print(y.shape)
        y = self.drop3(y)
        y = F.relu(self.conv2_2d(y))
        print(y.shape)
        #y = y.permute(0,2,3,1) 
        print(y.shape)
        outs = []
        outs.append(y)
        return outs
        

        return tuple(outs)
