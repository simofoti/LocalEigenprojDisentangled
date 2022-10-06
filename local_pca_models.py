import os
import argparse
import torch
import trimesh
import pickle
import numpy as np

from menpo.model import PCAModel
from menpo.shape import PointCloud
from torchvision.utils import make_grid, save_image
from pytorch3d.renderer import BlendParams

import utils
from model_manager import ModelManager
from data_generation_and_loading import get_data_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configurations/default.yaml',
                    help="Path to the configuration file.")
parser.add_argument('--id', type=str, default='none', help="ID of experiment")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

opts = parser.parse_args()
config = utils.get_config(opts.config)

n_components_local = 5

template = utils.load_template(config['data']['template_path'],
                               append_contours_to_feature=True)

train_loader, validation_loader, test_loader, normalization_dict = \
    get_data_loaders(config, template)

region_verts = {r: [] for r in template.feat_and_cont.keys()}
for data in train_loader:
    # un-normalize data
    mean_mesh = normalization_dict['mean'].to(data.x.device)
    std_mesh = normalization_dict['std'].to(data.x.device)
    if config['data']['swap_features']:
        bs = config['optimization']['batch_size']
        batch_diagonal_idx = [(bs + 1) * i for i in range(bs)]
        data.x = data.x[batch_diagonal_idx, ::]
    verts = data.x * std_mesh + mean_mesh

    # select vertices of each region and add them to local matrix
    for region, indices in template.feat_and_cont.items():
        region_verts[region].append(verts[:, indices['feature'], :])
        # optional - align vertices to template.

region_verts_tensors = {r: torch.cat(v, dim=0) for r, v in region_verts.items()}

local_pcas = {}
for region, batched_verts in region_verts_tensors.items():
    lst_verts = [PointCloud(batched_verts[i, ::].cpu().detach().numpy(), False)
                 for i in range(batched_verts.shape[0])]
    # Use 1/10 of data to avoid out of memory issues
    local_pcas[region] = PCAModel(lst_verts[:len(lst_verts) // 10],
                                  max_n_components=n_components_local)

all_vertices = []
for i in range(16):
    np_verts = np.zeros_like(template.pos.numpy())
    for region, pca in local_pcas.items():
        v_l = pca.instance(np.random.normal(size=n_components_local)).points
        np_verts[template.feat_and_cont[region]['feature'], :] = v_l
    # gen_mesh = trimesh.Trimesh(vertices=np_verts,
    #                            faces=template.face.numpy().T)
    # gen_mesh.show()
    all_vertices.append(torch.tensor(np_verts))

torch_vertices = torch.stack(all_vertices)

output_directory = os.path.join(opts.output_path + "/outputs", 'local_pca')
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

with open(os.path.join(output_directory, "vertices.pkl"), 'wb') as file:
    pickle.dump(torch_vertices, file)

if not torch.cuda.is_available():
    device = torch.device('cpu')
    print("GPU not available, running on CPU")
else:
    device = torch.device('cuda')

manager = ModelManager(
        configurations=config, device=device,
        precomputed_storage_path=config['data']['precomputed_path'])

blend_params = BlendParams(background_color=[1, 1, 1])
manager.default_shader.blend_params = blend_params
manager.simple_shader.blend_params = blend_params
manager.renderer.rasterizer.raster_settings.image_size = 512

renderings = manager.render(torch_vertices).cpu()
grid = make_grid(renderings, padding=10, pad_value=1)
save_image(grid, os.path.join(output_directory, 'random_generation.png'))

