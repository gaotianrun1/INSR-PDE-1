import os
import numpy as np
import torch
import torch.nn.functional as F
from base import sample_random, sample_uniform
from base import BaseModel, sample_random, sample_uniform
from .visualize import *
from .sampling import sample_mesh
import matplotlib.pyplot as plt
from .losses import *
from base import jacobian
from .torchgp import normalize, boundary_faces, sample_surface, volume_weighted_distribution, area_weighted_distribution, per_vertex_areas
import meshio

class ElasticityModel(BaseModel):
    # 继承自base中的baseModel,使用cfg文件进行配置
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.dim = cfg.dim

        # neural implicit network for deformation field
        self.deformation_field = self._create_network(self.dim, self.dim) # 当前位移场
        self.deformation_field_prev = self._create_network(self.dim, self.dim) # 上一时刻位移场
        self.deformation_field_prev_prev = self._create_network(self.dim, self.dim) # 上上时刻位移场
        self._set_require_grads(self.deformation_field_prev, False)
        self._set_require_grads(self.deformation_field_prev_prev, False)
        with torch.no_grad():
            self.deformation_field_prev.load_state_dict(self.deformation_field.state_dict())
            self.deformation_field_prev_prev.load_state_dict(self.deformation_field.state_dict())

        # initialize energy, deformable object and boundary conditions
        self._init_params(cfg)


    def _init_params(self, cfg):
        """
        根据cfg初始化模型参数
        """
        # basic config
        self.energy = cfg.energy
        self.use_mesh = cfg.use_mesh
        self.mesh_path = cfg.mesh_path
        self.sample_pattern = cfg.sample_pattern

        # config for the elastic energy
        self.ratio_arap = cfg.ratio_arap
        self.ratio_volume = cfg.ratio_volume
        self.ratio_kinematics = cfg.ratio_kinematics
        self.external_force_timesteps = cfg.external_force_timesteps

        # config for boundary conditions
        self.ratio_constraint = cfg.ratio_constraint
        self.ratio_collide = cfg.ratio_collide
        self.plane_height = cfg.plane_height
        self.circle_radius = cfg.collide_circle_radius
        self.external_force = torch.tensor([cfg.external_force_x, cfg.external_force_y, cfg.external_force_z])[0:self.dim].cuda()
        self.constraint_offset_right = torch.tensor([cfg.constraint_right_offset_x, cfg.constraint_right_offset_y, cfg.constraint_right_offset_z])[0:self.dim].cuda()
        self.circle_center = torch.tensor([cfg.collide_circle_x, cfg.collide_circle_y, cfg.collide_circle_z])[0:self.dim].cuda()
        
        # config for mesh setup (if use mesh)
        self.use_mesh = cfg.use_mesh
        if self.use_mesh == True:
            self._init_mesh()
        self.sample_vis = self._sample_in_visualization(self.vis_resolution)

        # config for initialization sampling (hyperparameters)
        if self.use_mesh:
            self.sample_resolution_init = self.sample_resolution
        else:
            if self.dim == 2:
                self.sample_resolution_init = 500
            elif self.dim == 3:
                self.sample_resolution_init = 100
            else:
                raise NotImplementedError


    def _init_mesh(self):
        # TODO：这里网格是要用来干什么？是INSR的计算方法之一，还是用纯网格计算作为另一种思路？
        # setup mesh
        self.mesh = meshio.read(self.mesh_path)
        self.mesh_V = torch.FloatTensor(self.mesh.points).cuda()
        if self.dim == 3:
            self.mesh_F = torch.tensor(self.mesh.cells_dict['tetra']).cuda()
            self.mesh_SF = torch.tensor(boundary_faces(self.mesh.cells_dict['tetra'])).cuda()
        elif self.dim == 2:
            self.mesh_F = torch.tensor(self.mesh.cells_dict['triangle']).cuda()
            self.mesh_SF = self.mesh_F
        # normalize the mesh
        self.mesh_V, self.mesh_F = normalize(self.mesh_V, self.mesh_F)
        self.mesh_V = self.mesh_V * 2.0
        self.vertex_area = per_vertex_areas(self.mesh_V, self.mesh_F)
        # precompute the area/volume distribution to accelerate the sampling
        if self.dim == 3:
            self.distrib = volume_weighted_distribution(self.mesh_V, self.mesh_F)
        elif self.dim == 2:
            self.distrib = area_weighted_distribution(self.mesh_V, self.mesh_F)


    @property
    def _trainable_networks(self):
        return {'deformation': self.deformation_field}


    # 时间步装饰器，表示 initialize 函数会在每个时间步被调用。在初始化时，将前两时刻的形变场权重设置为当前时刻的形变场。
    # @BaseModel._timestepping 是一个装饰器，用于对时间步的操作进行封装（例如增加时间步计数，保存检查点）
    @BaseModel._timestepping
    def initialize(self):
        self._initialize()
        # Initialize all the previous deformation fields to be zeros
        self.deformation_field_prev_prev.load_state_dict(self.deformation_field.state_dict())
        self.deformation_field_prev.load_state_dict(self.deformation_field.state_dict())


    @BaseModel._training_loop
    # 通过装饰器 @BaseModel._training_loop，完成了损失的优化和网络参数更新
    def _initialize(self):
        """
        initialize all field to zeros
        通过 _sample_in_training 采样初始化数据，计算形变场的输出并计算损失。
        """
        samples = self._sample_in_training(self.sample_resolution_init) # 初始化采样数量1000/100？

        out_wt = self.deformation_field(samples)
        loss_wt = torch.mean(out_wt ** 2)
        loss_dict = {'main': loss_wt}
        return loss_dict


    # 每个时间步后，更新前两时刻的形变场，并调用 _solve_deformation 进行形变求解。
    @BaseModel._timestepping
    def step(self):
        self.deformation_field_prev_prev.load_state_dict(self.deformation_field_prev.state_dict())
        self.deformation_field_prev.load_state_dict(self.deformation_field.state_dict())

        self._solve_deformation()
        
        
    # 通过这两个装饰器，ElasticityModel 的训练过程被解耦成了损失计算（在 model.py 中定义）和优化管理（在装饰器中封装）
    @BaseModel._training_loop
    def _solve_deformation(self):
        """
        采用n-2，n-1，n三个时刻的形变场神经表达，计算基于变分时间积分方法的能量损失函数
        通过优化损失函数来训练当前时刻的神经网络以表达形变场
        其中前两者是之前的训练结果，网络参数冻结不参与优化
        projection step for velocity: u <- u - grad(p)
        """

        samples = self._sample_in_training(self.sample_resolution)
        fixed_samples, fixed_samples_right = self._sample_fixed_in_training(self.sample_resolution)

        with torch.no_grad():
            q_prev = self.deformation_field_prev(samples) + samples # 应该是从形变算位置
            q_prev_prev = self.deformation_field_prev_prev(samples) + samples               
        q = self.deformation_field(samples) + samples
        
        # 通过样本计算当前时刻、前一时刻和上上时刻的形变，并通过时间步长 dt 计算速度（qdot）
        qdot = (q - q_prev) / self.dt
        qdot_prev = (q_prev - q_prev_prev) / self.dt
        
        # ARAP elasticity loss + kinematics loss + bc loss
        jac_x, _ = jacobian(q, samples) # (N, 2, 2) 描述了形变场在每个点的局部变化，元素Jij表示第i个坐标分量对第j个坐标分量的偏导
        U_x, S_x, V_x = torch.svd(jac_x)

        E_arap = self.ratio_arap * torch.sum((S_x - 1.0) ** 2) # 常用于形变建模的损失，基于每个点刚性变形假设，避免非物理的拉伸压缩
        E_volume = self.ratio_volume * torch.sum((torch.prod(S_x, dim=1) - 1) ** 2) # 惩罚导致物体体积变化的变形
        E_kinematics = self.ratio_kinematics * torch.sum((qdot - qdot_prev) ** 2) # 惩罚速度在不同时间步的变化，以保证运动平滑
        E_external = - self.dt * torch.sum(torch.mul(qdot, self.external_force.repeat(samples.shape[0], 1))) # 确保物体合理响应外力作用产生适当形变

        loss = 0
        # Energy for the elastodynamics equation 各部分能量项损失求和
        for l in self.energy:
            if l == 'arap':
                loss = loss + E_arap
            elif l == 'volume':
                loss = loss + E_volume
            elif l == 'kinematics':
                loss = loss + E_kinematics
            elif l == 'external':
                if self.timestep <= self.external_force_timesteps:
                    loss = loss + E_external
                else:  
                    loss = loss
            elif l == 'constraint':
                q_fixed = self.deformation_field(fixed_samples)
                E_constraint = positional_constraint_loss(q_fixed, 0, self.ratio_constraint)
                loss = loss + E_constraint
            elif l == 'constraint_right':
                q_fixed = self.deformation_field(fixed_samples_right)
                q_fixed_target = self.constraint_offset_right.repeat(fixed_samples_right.shape[0], 1)
                E_constraint_right = positional_constraint_loss(q_fixed, q_fixed_target, self.ratio_constraint)
                loss = loss + E_constraint_right
            elif l == 'constraint_right_compress':
                q_fixed = self.deformation_field(fixed_samples_right)
                q_fixed_target = -self.constraint_offset_right.repeat(fixed_samples_right.shape[0], 1)
                E_constraint_right_compress = positional_constraint_loss(q_fixed, q_fixed_target, self.ratio_constraint)
                loss = loss + E_constraint_right_compress
            elif l == 'collision':    # Collision force
                E_collide = collision_plane_loss(q, qdot, self.dt, self.ratio_collide, self.plane_height)
                loss = loss + E_collide
            elif l == 'collision_sphere':
                E_collide_sphere = collision_sphere_loss(q, qdot, self.dt, self.ratio_collide, self.circle_center, self.circle_radius)
                loss = loss + E_collide_sphere
            else:
                raise NotImplementedError

        loss_dict = {'main': loss}
        return loss_dict


    def _vis_solve_deformation(self):
        """形变场可视化"""
        fig = self.draw_field(self.vis_resolution, attr='deformation')
        self.tb.add_figure('stepU', fig, global_step=self.train_step)


    #################### sampling during training #######################
    def _sample_in_training(self, resolution):
        samples = []
        if self.use_mesh == True:
            for s in self.sample_pattern:
                if s == 'random':
                    random_samples = sample_mesh(self.mesh_V, self.mesh_F, resolution**self.dim, self.distrib)[:, 0:self.dim]
                    samples.append(random_samples.cuda().requires_grad_(True))
                elif s == 'uniform':
                    uniform_samples = self.mesh_V[:, 0:self.dim]
                    samples.append(uniform_samples.cuda().requires_grad_(True))
        else:
            for s in self.sample_pattern:
                if s == 'random':
                    random_samples = sample_random(resolution ** self.dim, self.dim, device=self.device).requires_grad_(True)
                    samples.append(random_samples)
                elif s == 'uniform':
                    uniform_samples = sample_uniform(resolution, self.dim, device=self.device).requires_grad_(True)
                    samples.append(uniform_samples)
                else:
                    raise NotImplementedError

        samples = torch.cat(samples, dim=0)
        return samples

    
    def _sample_fixed_in_training(self, resolution):
        # By default, the samples to be fixed is on the leftmost and rightmost side of square(2d) / cube (3d)
        # 左右边界上固定点的采样
        fixed_samples = []
        fixed_samples_right = []

        if self.use_mesh == False:
            for s in self.sample_pattern:
                if s == 'random':
                    random_fixed_samples = torch.cat((-torch.ones(resolution, 1, device=self.device), sample_random(resolution, self.dim-1, device=self.device)), dim=1)
                    fixed_samples.append(random_fixed_samples.cuda().requires_grad_(True))

                    random_fixed_samples_right = torch.cat((torch.ones(resolution, 1, device=self.device), sample_random(resolution, self.dim-1, device=self.device)), dim=1)
                    fixed_samples_right.append(random_fixed_samples_right.cuda().requires_grad_(True))
                elif s == 'uniform':
                    uniform_fixed_samples = torch.cat((-torch.ones(resolution**(self.dim-1), 1, device=self.device), sample_uniform(resolution, self.dim-1, device=self.device)), dim=1)
                    fixed_samples.append(uniform_fixed_samples.cuda().requires_grad_(True))

                    uniform_fixed_samples_right = torch.cat((torch.ones(resolution**(self.dim-1), 1, device=self.device), sample_uniform(resolution, self.dim-1, device=self.device)), dim=1)
                    fixed_samples_right.append(uniform_fixed_samples_right.cuda().requires_grad_(True))
                else:
                    raise NotImplementedError
            
        if len(fixed_samples) > 0:
            fixed_samples = torch.cat(fixed_samples, dim=0)
        if len(fixed_samples_right) > 0:
            fixed_samples_right = torch.cat(fixed_samples_right, dim=0)

        return fixed_samples, fixed_samples_right


################# visualization during training #####################

    def _sample_in_visualization(self, resolution, sample_boundary = True):
        # 为可视化生成采样点
        if self.use_mesh == True:
            samples = sample_surface(self.mesh_V, self.mesh_SF, resolution)[:, 0:self.dim]
            samples_V = self.mesh_V[:, 0:self.dim]
            samples = torch.vstack([samples, samples_V])
        else:
            samples = sample_uniform(resolution, self.dim, device=self.device)
            if sample_boundary == 1:
                # add uniform samples at fixed vertices
                fixed_samples = sample_uniform(resolution, self.dim-1, device=self.device)
                fixed_samples_left = torch.cat((-torch.ones(fixed_samples.shape[0], 1, device=self.device), fixed_samples), dim=1)
                samples = torch.vstack([samples, fixed_samples_left])
                
                fixed_samples_right = torch.cat((torch.ones(fixed_samples.shape[0], 1, device=self.device), fixed_samples), dim=1)
                samples = torch.vstack([samples, fixed_samples_right])
        return samples


    def _vis_deformation_field(self, resolution, attr="deformation"):
        pass


    def _sample_deformation_field(self, resolution=None, attr=None, to_numpy=True):
        # 采样形变场并计算采样点的新位置。
        if self.sample_vis is not None:
            samples = self.sample_vis
        else:
            samples =  self._sample_in_visualization(resolution)
        if attr == 'deformation':
            out = self.deformation_field(samples) + samples
        else:
            raise NotImplementedError
        if to_numpy:
            out = out.detach().cpu().numpy()
        return out


    def draw_field(self, resolution, attr=None, return_samples=False):
        # 绘制形变场的 2D 或 3D 图像
        sample_values = self._sample_deformation_field(resolution, attr, to_numpy=True)
        if attr == 'deformation':
            if self.dim == 2:
                if 'collision_sphere' in self.energy:
                    fig = draw_deformation_field2D(sample_values, color = sample_values[:,0] + sample_values[:,1], plane_height=self.plane_height, circle_center=self.circle_center, circle_radius=self.circle_radius)
                else:
                    fig = draw_deformation_field2D(sample_values, color = sample_values[:,0] + sample_values[:,1], plane_height=self.plane_height)
            elif self.dim == 3:
                if 'collision_sphere' in self.energy:
                    fig = draw_deformation_field3D(sample_values, color = sample_values[:,0] + sample_values[:,1] + sample_values[:,2], plane_height=self.plane_height, sphere_center=self.circle_center, sphere_radius=self.circle_radius)
                else:
                    fig = draw_deformation_field3D(sample_values, color = sample_values[:,0] + sample_values[:,1] + sample_values[:,2], plane_height=self.plane_height)
            if return_samples:
                return fig, sample_values
        else:
            raise NotImplementedError
        return fig


    def write_output(self, output_folder):
        fig, sample_values = self.draw_field(self.vis_resolution, attr='deformation', return_samples=True)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_deformation.png")
        save_figure(fig, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_deformation.ply")
        write_pointcloud_to_file(save_path, sample_values)
