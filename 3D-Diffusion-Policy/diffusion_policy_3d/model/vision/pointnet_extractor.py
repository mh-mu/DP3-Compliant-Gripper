import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from torchvision.models import resnet18

from diffusion_policy_3d.common.model_util import print_params

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
    

class DP3CompliantEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 use_compliant_image=False,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.img_key = 'combined_img'
        self.point_cloud_key = 'point_cloud'
        self.n_output_channels = out_channel
        
        # print('----')
        # print(observation_space)

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.img_shape = observation_space[self.img_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        cprint(f"[DP3CompliantEncoder] combined image shape: {self.img_shape}", "yellow")
        cprint(f"[DP3CompliantEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3CompliantEncoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_compliant_image = use_compliant_image
        
        # model for rgb image
        self.rgb_model = resnet18(pretrained=False)
        rgb_num_features = self.rgb_model.fc.in_features
        rgb_new_fc_layers = nn.Sequential(
            nn.Linear(rgb_num_features, self.n_output_channels)
        )
        self.rgb_model.fc = rgb_new_fc_layers
        

        if self.use_compliant_image:
            # model for compliant image
            self.compliant_model = resnet18(pretrained=False)
            compliant_num_features = self.compliant_model.fc.in_features
            compliant_new_fc_layers = nn.Sequential(
                nn.Linear(compliant_num_features, self.n_output_channels)
            )
            self.compliant_model.fc = compliant_new_fc_layers
            self.relu = nn.ReLU()
            self.fusion_fc = nn.Sequential(
                nn.Linear(self.n_output_channels*2, self.n_output_channels)
            )
           
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3CompliantEncoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        combined_img = observations[self.img_key].float()
        assert len(combined_img.shape) == 4, cprint(f"combined image shape: {combined_img.shape}, length should be 4", "red")
        
        # combined_img: B * 6 * H * W
        rgb_feat = self.rgb_model(combined_img[:, :3, :, :]) # B * out_channel
        if self.use_compliant_image:
            compliant_feat = self.compliant_model(combined_img[:, 3:, :, :]) # B * out_channel
            img_feat = self.relu(torch.cat([rgb_feat, compliant_feat], dim=-1))
            img_feat = self.fusion_fc(img_feat)
        else:
            img_feat = rgb_feat
            
        state = observations[self.state_key].float()
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([img_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels


class DP3PcdCompliantEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 use_compliant_image=False,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.img_key = 'combined_img'
        self.point_cloud_key = 'point_cloud'
        self.n_output_channels = out_channel
        
        print('----')
        print(observation_space)

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.img_shape = observation_space[self.img_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3CompliantEncoder] combined image shape: {self.img_shape}", "yellow")
        cprint(f"[DP3CompliantEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3CompliantEncoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_compliant_image = use_compliant_image
        
        # model for rgb image
        # self.rgb_model = resnet18(pretrained=False)
        # rgb_num_features = self.rgb_model.fc.in_features
        # rgb_new_fc_layers = nn.Sequential(
        #     nn.Linear(rgb_num_features, self.n_output_channels)
        # )
        # self.rgb_model.fc = rgb_new_fc_layers
        # print_params(self.rgb_model)

        self.rgb_model = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce the spatial dimensions
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.n_output_channels)  # Output vector of size n_output_channels
            )
        print_params(self.rgb_model)

        # model for point net 
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                print_params(self.extractor)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
                print_params(self.extractor)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # combine output of rgb image and pc
        self.fusion_fc = nn.Sequential(
                nn.Linear(self.n_output_channels*2, self.n_output_channels)
            )
        self.relu = nn.ReLU()

        if self.use_compliant_image:
            # model for compliant image
            # self.compliant_model = resnet18(pretrained=False)
            # compliant_num_features = self.compliant_model.fc.in_features
            # compliant_new_fc_layers = nn.Sequential(
            #     nn.Linear(compliant_num_features, self.n_output_channels)
            # )
            # self.compliant_model.fc = compliant_new_fc_layers
            self.fusion_fc = nn.Sequential(
                nn.Linear(self.n_output_channels*3, self.n_output_channels)
            )
            self.compliant_model = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce the spatial dimensions
                    nn.Flatten(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.n_output_channels)  # Output vector of size n_output_channels
                )
            print_params(self.compliant_model)
           
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        print_params(self.state_mlp)

        cprint(f"[DP3CompliantEncoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key].float()
        combined_img = observations[self.img_key].float()
        assert len(combined_img.shape) == 4, cprint(f"combined image shape: {combined_img.shape}, length should be 4", "red")
        pn_feat = self.extractor(points)    # B * out_channel
        # combined_img: B * 6 * H * W
        rgb_feat = self.rgb_model(combined_img[:, :3, :, :]) # B * out_channel

        if self.use_compliant_image:
            compliant_feat = self.compliant_model(combined_img[:, 3:, :, :]) # B * out_channel
            img_feat = self.relu(torch.cat([pn_feat, rgb_feat, compliant_feat], dim=-1))
            img_feat = self.fusion_fc(img_feat)
        else:
            img_feat = self.relu(torch.cat([pn_feat, rgb_feat], dim=-1))
            img_feat = self.fusion_fc(img_feat)
            
        state = observations[self.state_key].float()
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([img_feat, state_feat], dim=-1)
        # final_feat = torch.cat([pn_feat, state_feat], dim=-1) # debug
        return final_feat


    def output_shape(self):
        return self.n_output_channels