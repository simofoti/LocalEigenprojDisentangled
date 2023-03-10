import os
import pickle
import torch.nn
import trimesh
import time

import numpy as np

from sklearn import mixture
from torch.nn.functional import cross_entropy
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    BlendParams,
    HardGouraudShader
)

import utils
from mesh_simplification import MeshSimplifier
from compute_spirals import preprocess_spiral
from model import Model, FactorVAEDiscriminator


def get_model_manager(configurations, device, rendering_device=None,
                      precomputed_storage_path='precomputed'):
    if configurations['model_name'] in ['vae', 'ae']:
        manager = VaeManager
    elif configurations['model_name'] == 'sd_vae':
        manager = SdVaeManager
    elif configurations['model_name'] == 'led_vae':
        manager = LedVaeManager
    elif 'gan' in configurations['model_name']:
        manager = AllGansManager
    elif configurations['model_name'] == 'factor_vae':
        manager = FactorVaeManager
    elif configurations['model_name'] in ['dip_vae_i', 'dip_vae_ii']:
        manager = DipVaeManager
    elif configurations['model_name'] == 'rae':
        manager = RaeManager
    else:
        raise NotImplementedError

    return manager(configurations, device, rendering_device,
                   precomputed_storage_path)


class BaseManager(torch.nn.Module):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(BaseManager, self).__init__()
        self._model_params = configurations['model']
        self._optimization_params = configurations['optimization']
        self._precomputed_storage_path = precomputed_storage_path
        self._normalized_data = configurations['data']['normalize_data']
        self._epochs = self._optimization_params['epochs']

        self.train_set_length = None
        self.to_mm_const = configurations['data']['to_mm_constant']
        self.device = device
        self.template = utils.load_template(
            configurations['data']['template_path'])

        low_res_templates, down_transforms, up_transforms = \
            self._precompute_transformations()
        meshes_all_resolutions = [self.template] + low_res_templates
        spirals_indices = self._precompute_spirals(meshes_all_resolutions)

        self._latent_size = self._model_params['latent_size_id_regions'] *\
            len(self.template.feat_and_cont)

        self._losses = None
        self._w_kl_loss = float(self._optimization_params['kl_weight'])
        self._w_laplacian_loss = float(
            self._optimization_params['laplacian_weight'])

        self._swap_features = configurations['data']['swap_features']
        self._is_gan = False
        self.is_rae = False

        self._net = Model(in_channels=self._model_params['in_channels'],
                          out_channels=self._model_params['out_channels'],
                          latent_size=self._latent_size,
                          spiral_indices=spirals_indices,
                          down_transform=down_transforms,
                          up_transform=up_transforms,
                          pre_z_sigmoid=self._model_params['pre_z_sigmoid'],
                          is_vae=self._w_kl_loss > 0).to(device)

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=float(self._optimization_params['weight_decay']))

        self._latent_regions = self._compute_latent_regions()

        self._rend_device = rendering_device if rendering_device else device
        self.default_shader = HardGouraudShader(
            cameras=FoVPerspectiveCameras(),
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.simple_shader = ShadelessShader(
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.renderer = self._create_renderer()
        self._out_grid_size = 4

    @property
    def loss_keys(self):
        raise NotImplementedError

    @property
    def is_vae(self):
        return self._w_kl_loss > 0

    @property
    def is_gan(self):
        return self._is_gan

    @property
    def model_latent_size(self):
        return self._latent_size

    @property
    def latent_regions(self):
        return self._latent_regions

    def _precompute_transformations(self):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    'transforms.pkl')
        try:
            with open(storage_path, 'rb') as file:
                low_res_templates, down_transforms, up_transforms = \
                    pickle.load(file)
        except FileNotFoundError:
            print("Computing Down- and Up- sampling transformations ")
            if not os.path.isdir(self._precomputed_storage_path):
                os.mkdir(self._precomputed_storage_path)

            sampling_params = self._model_params['sampling']
            m = self.template

            r_weighted = False if sampling_params['type'] == 'basic' else True

            low_res_templates = []
            down_transforms = []
            up_transforms = []
            for sampling_factor in sampling_params['sampling_factors']:
                simplifier = MeshSimplifier(in_mesh=m, debug=False)
                m, down, up = simplifier(sampling_factor, r_weighted)
                low_res_templates.append(m)
                down_transforms.append(down)
                up_transforms.append(up)

            with open(storage_path, 'wb') as file:
                pickle.dump(
                    [low_res_templates, down_transforms, up_transforms], file)

        down_transforms = [d.to(self.device) for d in down_transforms]
        up_transforms = [u.to(self.device) for u in up_transforms]
        return low_res_templates, down_transforms, up_transforms

    def _precompute_spirals(self, templates):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    'spirals.pkl')
        try:
            with open(storage_path, 'rb') as file:
                spiral_indices_list = pickle.load(file)
        except FileNotFoundError:
            print("Computing Spirals")
            spirals_params = self._model_params['spirals']
            spiral_indices_list = []
            for i in range(len(templates) - 1):
                spiral_indices_list.append(
                    preprocess_spiral(templates[i].face.t().cpu().numpy(),
                                      spirals_params['length'][i],
                                      templates[i].pos.cpu().numpy(),
                                      spirals_params['dilation'][i]))
            with open(storage_path, 'wb') as file:
                pickle.dump(spiral_indices_list, file)
        spiral_indices_list = [s.to(self.device) for s in spiral_indices_list]
        return spiral_indices_list

    def _compute_latent_regions(self):
        region_names = list(self.template.feat_and_cont.keys())
        region_size = self._model_params['latent_size_id_regions']
        latent_regions = {k: [i * region_size, (i + 1) * region_size]
                          for i, k in enumerate(region_names)}
        return latent_regions

    def forward(self, data):
        return self._net(data.x)

    @torch.no_grad()
    def encode(self, x):
        self._net.eval()
        return self._net.encode(x)[0]

    @torch.no_grad()
    def generate(self, z):
        self._net.eval()
        return self._net.decode(z)

    def generate_for_opt(self, z):
        self._net.train()
        return self._net.decode(z)

    def run_epoch(self, data_loader, device, train=True):
        if train:
            self._net.train()
        else:
            self._net.eval()

        self._reset_losses()
        it = 0
        for it, data in enumerate(data_loader):
            if train:
                losses = self._do_iteration(data, device, train=True)
            else:
                with torch.no_grad():
                    losses = self._do_iteration(data, device, train=False)
            self._add_losses(losses)
        self._divide_losses(it + 1)

    def _do_iteration(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _compute_l1_loss(prediction, gt, reduction='mean'):
        return torch.nn.L1Loss(reduction=reduction)(prediction, gt)

    @staticmethod
    def compute_mse_loss(prediction, gt, reduction='mean'):
        return torch.nn.MSELoss(reduction=reduction)(prediction, gt)

    def _compute_laplacian_regularizer(self, prediction):
        bs = prediction.shape[0]
        n_verts = prediction.shape[1]
        laplacian = self.template.laplacian.to(prediction.device)
        prediction_laplacian = utils.batch_mm(laplacian, prediction)
        loss = prediction_laplacian.norm(dim=-1) / n_verts
        return loss.sum() / bs

    @staticmethod
    def _compute_kl_divergence_loss(mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kl, dim=0)

    @staticmethod
    def _compute_embedding_loss(z):
        return (z ** 2).mean(dim=1)

    def compute_vertex_errors(self, out_verts, gt_verts):
        vertex_errors = self.compute_mse_loss(
            out_verts, gt_verts, reduction='none')
        vertex_errors = torch.sqrt(torch.sum(vertex_errors, dim=-1))
        vertex_errors *= self.to_mm_const
        return vertex_errors

    def _reset_losses(self):
        self._losses = {k: 0 for k in self.loss_keys}

    def _add_losses(self, additive_losses):
        for k in self.loss_keys:
            loss = additive_losses[k]
            self._losses[k] += loss.item() if torch.is_tensor(loss) else loss

    def _divide_losses(self, value):
        for k in self.loss_keys:
            self._losses[k] /= value

    def log_losses(self, writer, epoch, phase='train'):
        for k in self.loss_keys:
            loss = self._losses[k]
            loss = loss.item() if torch.is_tensor(loss) else loss
            writer.add_scalar(
                phase + '/' + str(k), loss, epoch + 1)

    def log_images(self, in_data, writer, epoch, normalization_dict=None,
                   phase='train', error_max_scale=5):
        gt_meshes = in_data.x.to(self._rend_device)
        out_meshes = self.forward(in_data.to(self.device))[0]
        out_meshes = out_meshes.to(self._rend_device)

        if self._normalized_data:
            mean_mesh = normalization_dict['mean'].to(self._rend_device)
            std_mesh = normalization_dict['std'].to(self._rend_device)
            gt_meshes = gt_meshes * std_mesh + mean_mesh
            out_meshes = out_meshes * std_mesh + mean_mesh

        vertex_errors = self.compute_vertex_errors(out_meshes, gt_meshes)

        gt_renders = self.render(gt_meshes)
        out_renders = self.render(out_meshes)
        errors_renders = self.render(out_meshes, vertex_errors,
                                     error_max_scale)
        log = torch.cat([gt_renders, out_renders, errors_renders], dim=-1)

        log = make_grid(log, padding=10, pad_value=1, nrow=self._out_grid_size)
        writer.add_image(tag=phase, global_step=epoch + 1, img_tensor=log)

    def _create_renderer(self, img_size=256):
        raster_settings = RasterizationSettings(image_size=img_size)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings,
                                      cameras=FoVPerspectiveCameras()),
            shader=self.default_shader)
        renderer.to(self._rend_device)
        return renderer

    @torch.no_grad()
    def render(self, batched_data, vertex_errors=None, error_max_scale=None):
        batch_size = batched_data.shape[0]
        batched_verts = batched_data.detach().to(self._rend_device)
        template = self.template.to(self._rend_device)

        if vertex_errors is not None:
            self.renderer.shader = self.simple_shader
            textures = TexturesVertex(utils.errors_to_colors(
                vertex_errors, min_value=0,
                max_value=error_max_scale, cmap='plasma') / 255)
        else:
            self.renderer.shader = self.default_shader
            textures = TexturesVertex(torch.ones_like(batched_verts) * 0.5)

        meshes = Meshes(
            verts=batched_verts,
            faces=template.face.t().expand(batch_size, -1, -1),
            textures=textures)

        rotation, translation = look_at_view_transform(
            dist=2.5, elev=0, azim=15)
        cameras = FoVPerspectiveCameras(R=rotation, T=translation,
                                        device=self._rend_device)

        lights = PointLights(location=[[0.0, 0.0, 3.0]],
                             diffuse_color=[[1., 1., 1.]],
                             device=self._rend_device)

        materials = Materials(shininess=0.5, device=self._rend_device)

        images = self.renderer(meshes, cameras=cameras, lights=lights,
                               materials=materials).permute(0, 3, 1, 2)
        return images[:, :3, ::]

    def render_and_show_batch(self, data, normalization_dict):
        verts = data.x.to(self._rend_device)
        if self._normalized_data:
            mean_mesh = normalization_dict['mean'].to(self._rend_device)
            std_mesh = normalization_dict['std'].to(self._rend_device)
            verts = verts * std_mesh + mean_mesh
        rend = self.render(verts)
        grid = make_grid(rend, padding=10, pad_value=1,
                         nrow=self._out_grid_size)
        img = ToPILImage()(grid)
        img.show()

    def show_mesh(self, vertices, normalization_dict=None):
        vertices = torch.squeeze(vertices)
        if self._normalized_data:
            mean_verts = normalization_dict['mean'].to(vertices.device)
            std_verts = normalization_dict['std'].to(vertices.device)
            vertices = vertices * std_verts + mean_verts
        mesh = trimesh.Trimesh(vertices.cpu().detach().numpy(),
                               self.template.face.t().cpu().numpy())
        mesh.show()

    def save_weights(self, checkpoint_dir, epoch):
        net_name = os.path.join(checkpoint_dir, 'model_%08d.pt' % (epoch + 1))
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save({'model': self._net.state_dict()}, net_name)
        torch.save({'optimizer': self._optimizer.state_dict()}, opt_name)

    def resume(self, checkpoint_dir):
        last_model_name = utils.get_model_list(checkpoint_dir, 'model')
        state_dict = torch.load(last_model_name)
        self._net.load_state_dict(state_dict['model'])
        epochs = int(last_model_name[-11:-3])
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self._optimizer.load_state_dict(state_dict['optimizer'])
        print(f"Resume from epoch {epochs}")
        return epochs

    def initialize_local_eigenvectors(self, *args, **kwargs):
        pass


class VaeManager(BaseManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(VaeManager, self).__init__(configurations, device,
                                         rendering_device,
                                         precomputed_storage_path)
        if configurations['model_name'] == 'ae':
            assert self._w_kl_loss <= 0

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl', 'laplacian', 'tot']

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer.zero_grad()

        data = data.to(device)
        reconstructed, z, mu, logvar = self.forward(data)
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)

        if self._w_kl_loss > 0:
            loss_kl = self._compute_kl_divergence_loss(mu, logvar)
        else:
            loss_kl = torch.tensor(0, device=device)

        loss_tot = loss_recon + self._w_kl_loss * loss_kl + \
            self._w_laplacian_loss * loss_laplacian

        if train:
            loss_tot.backward()
            self._optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'laplacian': loss_laplacian.item(),
                'tot': loss_tot.item()}


class SdVaeManager(BaseManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(SdVaeManager, self).__init__(configurations, device,
                                           rendering_device,
                                           precomputed_storage_path)
        self._w_latent_cons_loss = float(
            self._optimization_params['latent_consistency_weight'])

        bs = self._optimization_params['batch_size']
        self._out_grid_size = bs
        self._batch_diagonal_idx = [(bs + 1) * i for i in range(bs)]
        assert self._w_kl_loss > 0 and self._w_latent_cons_loss > 0
        assert self._swap_features

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl', 'latent_consistency',
                'laplacian', 'tot']

    @property
    def batch_diagonal_idx(self):
        return self._batch_diagonal_idx

    def _compute_latent_consistency(self, z, swapped_feature):
        bs = self._optimization_params['batch_size']
        eta1 = self._optimization_params['latent_consistency_eta1']
        eta2 = self._optimization_params['latent_consistency_eta2']
        latent_region = self._latent_regions[swapped_feature]
        z_feature = z[:, latent_region[0]:latent_region[1]].view(bs, bs, -1)
        z_else = torch.cat([z[:, :latent_region[0]],
                            z[:, latent_region[1]:]], dim=1).view(bs, bs, -1)
        triu_indices = torch.triu_indices(
            z_feature.shape[0], z_feature.shape[0], 1)

        lg = z_feature.unsqueeze(0) - z_feature.unsqueeze(1)
        lg = lg[triu_indices[0], triu_indices[1], :, :].reshape(-1,
                                                                lg.shape[-1])
        lg = torch.sum(lg ** 2, dim=-1)

        dg = z_feature.permute(1, 2, 0).unsqueeze(0) - \
            z_feature.permute(1, 2, 0).unsqueeze(1)
        dg = dg[triu_indices[0], triu_indices[1], :, :].permute(0, 2, 1)
        dg = torch.sum(dg.reshape(-1, dg.shape[-1]) ** 2, dim=-1)

        dr = z_else.unsqueeze(0) - z_else.unsqueeze(1)
        dr = dr[triu_indices[0], triu_indices[1], :, :].reshape(-1,
                                                                dr.shape[-1])
        dr = torch.sum(dr ** 2, dim=-1)

        lr = z_else.permute(1, 2, 0).unsqueeze(0) - \
            z_else.permute(1, 2, 0).unsqueeze(1)
        lr = lr[triu_indices[0], triu_indices[1], :, :].permute(0, 2, 1)
        lr = torch.sum(lr.reshape(-1, lr.shape[-1]) ** 2, dim=-1)
        zero = torch.tensor(0, device=z.device)
        return (1 / (bs ** 3 - bs ** 2)) * \
               (torch.sum(torch.max(zero, lr - dr + eta2)) +
                torch.sum(torch.max(zero, lg - dg + eta1)))

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer.zero_grad()

        data = data.to(device)
        reconstructed, z, mu, logvar = self.forward(data)
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)
        loss_kl = self._compute_kl_divergence_loss(mu, logvar)
        loss_z_cons = self._compute_latent_consistency(z, data.swapped)

        loss_tot = loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_latent_cons_loss * loss_z_cons + \
            self._w_laplacian_loss * loss_laplacian

        if train:
            loss_tot.backward()
            self._optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'latent_consistency': loss_z_cons.item(),
                'laplacian': loss_laplacian.item(),
                'tot': loss_tot.item()}


class DipVaeManager(BaseManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(DipVaeManager, self).__init__(configurations, device,
                                            rendering_device,
                                            precomputed_storage_path)
        self._w_dip_loss = float(self._optimization_params['dip_weight'])
        assert not self._swap_features
        assert self._w_kl_loss > 0 and self._w_dip_loss > 0

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl', 'dip', 'laplacian', 'tot']

    def _compute_dip_loss(self, mu, logvar):
        centered_mu = mu - mu.mean(dim=1, keepdim=True)
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze()

        if self._optimization_params['dip_type'] == 'ii':
            cov_z = cov_mu + torch.mean(
                torch.diagonal((2. * logvar).exp(), dim1=0), dim=0)
        else:
            cov_z = cov_mu

        cov_diag = torch.diag(cov_z)
        cov_offdiag = cov_z - torch.diag(cov_diag)

        lambda_diag = self._optimization_params['dip_diag_lambda']
        lambda_offdiag = self._optimization_params['dip_offdiag_lambda']
        return lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
            lambda_diag * torch.sum((cov_diag - 1) ** 2)

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer.zero_grad()

        data = data.to(device)
        reconstructed, z, mu, logvar = self.forward(data)
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)

        loss_kl = self._compute_kl_divergence_loss(mu, logvar)

        if self._w_dip_loss > 0:
            loss_dip = self._compute_dip_loss(mu, logvar)
        else:
            loss_dip = torch.tensor(0, device=device)

        loss_tot = loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_dip_loss * loss_dip + \
            self._w_laplacian_loss * loss_laplacian

        if train:
            loss_tot.backward()
            self._optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'dip': loss_dip.item(),
                'laplacian': loss_laplacian.item(),
                'tot': loss_tot.item()}


class FactorVaeManager(BaseManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(FactorVaeManager, self).__init__(configurations, device,
                                               rendering_device,
                                               precomputed_storage_path)
        self._w_factor_loss = float(self._optimization_params['factor_weight'])
        assert not self._swap_features
        assert self._w_kl_loss > 0 and self._w_factor_loss > 0
        self._factor_discriminator = FactorVAEDiscriminator(
            self._latent_size).to(device)
        self._disc_optimizer = torch.optim.Adam(
            self._factor_discriminator.parameters(),
            lr=float(self._optimization_params['lr']), betas=(0.5, 0.9),
            weight_decay=float(self._optimization_params['weight_decay']))

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl', 'factor', 'laplacian', 'tot']

    @staticmethod
    def _permute_latent_dims(latent_sample):
        perm = torch.zeros_like(latent_sample)
        batch_size, dim_z = perm.size()
        for z in range(dim_z):
            pi = torch.randperm(batch_size).to(latent_sample.device)
            perm[:, z] = latent_sample[pi, z]
        return perm

    def _do_iteration(self, data, device='cpu', train=True):
        # Factor-vae split data into two batches.
        data = data.to(device)
        batch_size = data.x.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.x.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        reconstructed1, z1, mu1, logvar1 = self._net(data1)
        loss_recon = self.compute_mse_loss(reconstructed1, data1)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed1)

        loss_kl = self._compute_kl_divergence_loss(mu1, logvar1)

        disc_z = self._factor_discriminator(z1)
        factor_loss = (disc_z[:, 0] - disc_z[:, 1]).mean()

        loss_tot = loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_laplacian_loss * loss_laplacian + \
            self._w_factor_loss * factor_loss

        if train:
            self._optimizer.zero_grad()
            loss_tot.backward(retain_graph=True)

            _, z2, _, _ = self._net(data2)
            z2_perm = self._permute_latent_dims(z2).detach()
            disc_z_perm = self._factor_discriminator(z2_perm)
            ones = torch.ones(half_batch_size, dtype=torch.long,
                              device=self.device)
            zeros = torch.zeros_like(ones)
            disc_factor_loss = 0.5 * (cross_entropy(disc_z, zeros) +
                                      cross_entropy(disc_z_perm, ones))

            self._disc_optimizer.zero_grad()
            disc_factor_loss.backward()
            self._optimizer.step()
            self._disc_optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'factor': factor_loss.item(),
                'laplacian': loss_laplacian.item(),
                'tot': loss_tot.item()}


class RaeManager(BaseManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(RaeManager, self).__init__(configurations, device,
                                         rendering_device,
                                         precomputed_storage_path)
        self.is_rae = True
        self._w_rae_loss = float(self._optimization_params['rae_weight'])
        assert self._w_kl_loss == 0
        assert self._w_dip_loss == 0 and self._w_factor_loss == 0

        self._gaussian_mixture = None
        enc_w_decay = self._optimization_params['weight_decay']

        if float(self._optimization_params['rae_grad_penalty']) > 0:
            gen_w_decay = enc_w_decay
        else:
            gen_w_decay = self._optimization_params['rae_gen_weight_decay']

        self._optimizer = torch.optim.Adam(
            self._net.en_layers.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=float(enc_w_decay))
        self._rae_gen_optimizer = torch.optim.Adam(
            self._net.de_layers.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=float(gen_w_decay))

    @property
    def loss_keys(self):
        return ['reconstruction', 'rae', 'laplacian', 'tot']

    @staticmethod
    def _compute_gradient_penalty_loss(z, prediction):
        grads = torch.autograd.grad(prediction ** 2, z,
                                    grad_outputs=torch.ones_like(prediction),
                                    create_graph=True, retain_graph=True)[0]
        return torch.mean(grads ** 2, dim=1)

    def _compute_rae_loss(self, z, prediction):
        rae_embedding = float(self._optimization_params['rae_embedding'])  # 0.5
        rae_grad_penalty = float(self._optimization_params['rae_grad_penalty'])

        rae_loss = rae_embedding * self._compute_embedding_loss(z)

        if rae_grad_penalty > 0:
            rae_loss += rae_grad_penalty * \
                        self._compute_gradient_penalty_loss(z, prediction)
        return rae_loss.mean()

    def run_epoch(self, data_loader, device, train=True):
        if train:
            self._net.train()
        else:
            self._net.eval()

        self._reset_losses()
        it = 0
        for it, data in enumerate(data_loader):
            if train:
                losses = self._do_iteration(data, device, train=True)
            else:
                losses = self._do_iteration(data, device, train=False)
            self._add_losses(losses)
        self._divide_losses(it + 1)

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer.zero_grad()
            self._rae_gen_optimizer.zero_grad()

        data = data.to(device)
        reconstructed, z, mu, logvar = self.forward(data)
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)
        loss_rae = self._compute_rae_loss(z, reconstructed)

        loss_tot = loss_recon + \
            self._w_rae_loss * loss_rae + \
            self._w_laplacian_loss * loss_laplacian

        if train:
            loss_tot.backward()
            self._optimizer.step()
            self._rae_gen_optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'rae': loss_rae.item(),
                'laplacian': loss_laplacian.item(),
                'tot': loss_tot.item()}

    @torch.no_grad()
    def fit_gaussian_mixture(self, train_loader):
        latents_list = []
        for data in train_loader:
            if self._swap_features:
                x = data.x[self._batch_diagonal_idx, ::]
            else:
                x = data.x
            latents_list.append(self.encode(x.to(self.device)).detach().cpu())
        latents = torch.cat(latents_list, dim=0)

        gmm = mixture.GaussianMixture(
            n_components=self._optimization_params['rae_n_gaussians'],
            means_init=None,  # TODO: use labels (if available) to compute means
            covariance_type="full", max_iter=2000, verbose=0, tol=1e-3)
        gmm.fit(latents.cpu().detach())
        self._gaussian_mixture = gmm

    def sample_gaussian_mixture(self, n_samples):
        z = self._gaussian_mixture.sample(n_samples)[0]
        return torch.tensor(z, device=self.device, dtype=torch.float)

    def score_samples_gaussian_mixture(self, samples):
        return self._gaussian_mixture.score_samples(samples)

    def save_weights(self, checkpoint_dir, epoch):
        super().save_weights(checkpoint_dir, epoch)
        gmm_name = os.path.join(checkpoint_dir,
                                'gmm_%08d.pkl' % (epoch + 1))
        with open(gmm_name, 'wb') as f:
            pickle.dump(self._gaussian_mixture, f)

    def resume(self, checkpoint_dir):
        last_model_name = utils.get_model_list(checkpoint_dir, 'model')
        epochs = super().resume(checkpoint_dir)
        if self.is_rae:
            gmm_name = last_model_name.replace('model', 'gmm')
            with open(gmm_name, 'rb') as f:
                self._gaussian_mixture = pickle.load(f)
        print(f"Resume from epoch {epochs}")
        return epochs


class LedModelsManager(BaseManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(LedModelsManager, self).__init__(configurations, device,
                                               rendering_device,
                                               precomputed_storage_path)
        self._w_lep_loss = float(
            self._optimization_params['local_eigenprojection_weight'])

        if self._w_lep_loss > 0:
            start_time = time.time()
            self._local_eigenvectors = utils.compute_local_eigenvectors(
                template=self.template,
                k=int(self._optimization_params['local_eigendecomposition_k']))
            self._w_lep_gen_loss = float(
                self._optimization_params['local_eigenprojection_gen_weight'])
            self._verts_std = None
            self._local_ep_means, self._local_ep_stds = None, None
            end_time = time.time()
            print(f"Process time eigenvectors computation = "
                  f"{end_time - start_time}s")

    @property
    def loss_keys(self):
        raise NotImplementedError

    def initialize_local_eigenvectors(self, train_loader, normalization_dict,
                                      plot_eigenproj_distributions=False):
        start_time = time.time()
        if self._normalized_data:  # important for projection
            self._verts_std = normalization_dict['std']

        # Initial eigenvectors are associated to low frequencies. Also, the
        # first one has an associated eigenvalue of 0 and the second is
        # describing the order of the vertices in the mesh (Fiedler vector).
        # It may be useful to exclude the first n eigenvectors.
        initial_ev = self._optimization_params[
            'local_eigenvectors_remove_first_n']
        for k, local_evs in self._local_eigenvectors.items():
            self._local_eigenvectors[k] = local_evs[:, initial_ev:]

        local_eps = {k: [] for k in self._local_eigenvectors.keys()}
        for data in train_loader:
            local_ep = self._local_eigenproject_sigend_distances(data.x)
            for k in local_eps.keys():
                local_eps[k].append(local_ep[k])

        local_ep_means = {k: torch.mean(torch.cat(epm, dim=0), dim=0)
                          for k, epm in local_eps.items()}
        local_ep_stds = {k: torch.std(torch.cat(epm, dim=0), dim=0)
                         for k, epm in local_eps.items()}

        if self._optimization_params['local_eigenprojection_max_variance']:
            order_variance = {k: np.argsort(- eps.detach().cpu().numpy())
                              for k, eps in local_ep_stds.items()}
            for k, order in order_variance.items():
                self._local_eigenvectors[k] = \
                    self._local_eigenvectors[k][:, order]
                local_ep_means[k] = local_ep_means[k][order]
                local_ep_stds[k] = local_ep_stds[k][order]

        # select only relevant eigenvectors and make sure everything
        # that is needed for eigenprojection is on the correct device
        for k, local_evs in self._local_eigenvectors.items():
            local_latent_size = self._latent_regions[k][1] - \
                                self._latent_regions[k][0]
            local_evs = torch.tensor(local_evs, device=self.device)
            self._local_eigenvectors[k] = local_evs[:, :local_latent_size]
            local_ep_means[k] = \
                local_ep_means[k][:local_latent_size].to(self.device)
            local_ep_stds[k] = \
                local_ep_stds[k][:local_latent_size].to(self.device)
        self._local_ep_means = local_ep_means
        self._local_ep_stds = local_ep_stds
        self.template.norm = self.template.norm.to(self.device)
        self.template.pos = self.template.pos.to(self.device)
        if self._normalized_data:
            self._verts_std = self._verts_std.to(self.device)

        end_time = time.time()
        print(f"Process time initialize LEP = {end_time - start_time}")

        if plot_eigenproj_distributions:
            all_ep = []
            for data in train_loader:
                x = data.x.to(self.device)
                local_ep = self._local_eigenproject_sigend_distances(x)
                ep_means = torch.cat(list(self._local_ep_means.values()))
                ep_stds = torch.cat(list(self._local_ep_stds.values()))
                n_local_ep = (torch.cat(list(local_ep.values()), dim=1) -
                              ep_means) / ep_stds
                all_ep.append(n_local_ep)
            utils.plot_eigproj(
                torch.cat(all_ep, dim=0),
                colors_as_str=list(self.template.feat_and_cont.keys()))

    def _local_eigenproject_sigend_distances(self, x):
        sd = utils.compute_signed_distances(x, self.template, self._verts_std)
        projections = {}
        for k, u_local in self._local_eigenvectors.items():
            region_selector = self.template.feat_and_cont[k]['feature']
            local_projection = torch.tensor(u_local.T) @ sd[:, region_selector]
            projections[k] = local_projection.squeeze(dim=-1)
        return projections

    def _local_eigenproject(self, sd, region):
        region_selector = self.template.feat_and_cont[region]['feature']
        u_local = self._local_eigenvectors[region]
        local_projection = u_local.t() @ sd[:, region_selector]
        return local_projection.squeeze(dim=-1)

    def _compute_local_eigenprojection_loss(self, z, x):
        sd = utils.compute_signed_distances(x, self.template,
                                            self._verts_std)
        lep_loss = torch.zeros(x.shape[0], device=self.device)
        for k, z_local_range in self._latent_regions.items():
            local_sd_ep = self._local_eigenproject(sd, k)
            norm_local_sd_ep = \
                (local_sd_ep - self._local_ep_means[k]) / self._local_ep_stds[k]
            lep_loss += torch.mean(torch.abs(
                z[:, z_local_range[0]:z_local_range[1]] - norm_local_sd_ep),
                dim=1)
        return lep_loss.mean()

    def _do_iteration(self, *args, **kwargs):
        raise NotImplementedError


class LedVaeManager(LedModelsManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(LedVaeManager, self).__init__(configurations, device,
                                            rendering_device,
                                            precomputed_storage_path)
        assert self._w_lep_loss > 0

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl', 'local_eigenprojection',
                'local_eigenprojection_gen', 'laplacian', 'tot']

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer.zero_grad()

        data = data.to(device)
        reconstructed, z, mu, logvar = self.forward(data)
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)
        loss_kl = self._compute_kl_divergence_loss(mu, logvar)

        loss_local_eigenproj = self._compute_local_eigenprojection_loss(
            mu, data.x.detach())

        if self._w_lep_gen_loss > 0:
            loss_local_eigenproj_gen = self._compute_local_eigenprojection_loss(
                mu.detach(), reconstructed)
            loss_local_eigenproj_gen_weighted = loss_local_eigenproj_gen * \
                                                self._w_lep_gen_loss
        else:
            loss_local_eigenproj_gen = torch.tensor(0, device=device)
            loss_local_eigenproj_gen_weighted = torch.tensor(0, device=device)

        loss_tot = loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_lep_loss * loss_local_eigenproj + \
            self._w_laplacian_loss * loss_laplacian

        if train:
            if self._w_lep_gen_loss > 0:
                loss_local_eigenproj_gen_weighted.backward(retain_graph=True)
                self._net.en_layers.zero_grad()
            loss_tot.backward()
            self._optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'local_eigenprojection': loss_local_eigenproj.item(),
                'local_eigenprojection_gen': loss_local_eigenproj_gen.item(),
                'laplacian': loss_laplacian.item(),
                'tot': loss_tot.item()}


class AllGansManager(LedModelsManager):
    def __init__(self, configurations, device, rendering_device,
                 precomputed_storage_path):
        super(AllGansManager, self).__init__(configurations, device,
                                             rendering_device,
                                             precomputed_storage_path)
        self._is_gan = True
        assert not self._swap_features
        assert self._w_kl_loss <= 0
        assert not self._model_params['pre_z_sigmoid']
        self._iteration_counter = 0
        self._gan_type = self._optimization_params['gan_type']

        if self._gan_type == 'wgan':
            gen_opt = torch.optim.RMSprop
            disc_opt = torch.optim.RMSprop
            self._disc_terminal_layer = torch.nn.Sequential(
                torch.nn.ELU(),
                torch.nn.Linear(self._latent_size, 1)
            ).to(device)
            self._disc_params = [*self._net.en_layers.parameters(),
                                 *self._disc_terminal_layer.parameters()]
        else:
            gen_opt = torch.optim.Adam
            disc_opt = torch.optim.SGD
            self._disc_params = self._net.en_layers.parameters()
        self._optimizer = gen_opt(
            self._net.de_layers.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=float(self._optimization_params['weight_decay']))
        self._disc_optimizer = disc_opt(
            self._disc_params,
            lr=float(self._optimization_params['gan_disc_lr']),
            weight_decay=float(self._optimization_params['weight_decay']))

    @property
    def loss_keys(self):
        return ['gen', 'disc', 'local_eigenprojection', 'laplacian', 'tot']

    def _do_iteration(self, data, device='cpu', train=True):
        self._iteration_counter += 1 if train else 0
        self._optimizer.zero_grad()

        data = data.to(device)
        z = torch.randn([data.x.shape[0], self.model_latent_size],
                        device=device)

        x_real = data.x
        x_fake = self._net.decode(z)
        loss_laplacian = self._compute_laplacian_regularizer(x_fake)

        if self._w_lep_loss > 0:
            loss_local_eigenproj = self._compute_local_eigenprojection_loss(
                z.detach(), x_fake)
        else:
            loss_local_eigenproj = torch.tensor(0, device=device)

        if self._optimization_params['gan_noise_anneal_length_percentage'] > 0:
            noise_std = utils.annealing_coefficient(
                self._iteration_counter,
                self._epochs * self.train_set_length,
                self._optimization_params['gan_noise_anneal_length_percentage'])
            x_real += noise_std * torch.randn_like(x_real)
            x_fake += noise_std * torch.randn_like(x_fake)

        loss_gen = self._generator_loss(x_fake)

        loss_tot = loss_gen + self._w_lep_loss * loss_local_eigenproj + \
            self._w_laplacian_loss * loss_laplacian

        if train:
            loss_tot.backward()
            self._optimizer.step()

        # Train Discriminator
        self._disc_optimizer.zero_grad()
        loss_disc = self._discriminator_loss(x_real=x_real, x_fake=x_fake)

        if train and self._iteration_counter % \
                self._optimization_params['gan_disc_train_every'] == 0:
            loss_disc.backward()
            self._disc_optimizer.step()

        return {'gen': loss_gen.item(),
                'disc': loss_disc.item(),
                'local_eigenprojection': loss_local_eigenproj.item(),
                'laplacian': loss_laplacian.item(),
                'tot': loss_tot.item()}

    def _discriminator_loss(self, x_real, x_fake):
        if self._gan_type == 'lsgan':
            out_real = self._net.encode(x_real)[0]
            out_fake = self._net.encode(x_fake.detach())[0]
            loss = torch.mean((out_real - 1) ** 2) + torch.mean(out_fake ** 2)
            loss /= 2
        else:  # wgan
            out_real = self._net.encode(x_real)[0]
            out_real = self._disc_terminal_layer(out_real)
            out_fake = self._net.encode(x_fake.detach())[0]
            out_fake = self._disc_terminal_layer(out_fake)
            loss = - torch.mean(out_real) + torch.mean(out_fake)
            for p in self._disc_params:
                p.data.clamp_(-0.01, 0.01)
        return loss

    def _generator_loss(self, x_fake):
        if self._gan_type == 'lsgan':
            out_fake = self._net.encode(x_fake)[0]
            loss = torch.mean((out_fake - 1) ** 2) / 2
        else:  # wgan
            out_fake = self._net.encode(x_fake)[0]
            out_fake = self._disc_terminal_layer(out_fake)
            loss = - torch.mean(out_fake)
        return loss

    def log_images(self, in_data, writer, epoch, normalization_dict=None,
                   phase='train', error_max_scale=5):
        z = torch.randn([in_data.x.shape[0], self.model_latent_size],
                        device=self._rend_device)
        x_gen = self.generate(z)
        if self._normalized_data:
            mean_mesh = normalization_dict['mean'].to(self._rend_device)
            std_mesh = normalization_dict['std'].to(self._rend_device)
            x_gen = x_gen * std_mesh + mean_mesh
        log = self.render(x_gen)

        log = make_grid(log, padding=10, pad_value=1, nrow=self._out_grid_size)
        writer.add_image(tag=phase, global_step=epoch + 1, img_tensor=log)


class ShadelessShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = \
            blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        pixel_colors = meshes.sample_textures(fragments)
        images = hard_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images
