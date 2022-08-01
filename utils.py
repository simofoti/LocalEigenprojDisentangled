import os
import yaml
import trimesh
import torch
import joypy

import matplotlib.cm
import torch_geometric.transforms

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.utils import subgraph
from scipy.sparse.linalg import eigsh
from matplotlib.colors import ListedColormap


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print(f"Creating directory: {checkpoint_directory}")
        os.makedirs(checkpoint_directory)
    return checkpoint_directory


def load_template(mesh_path, append_contours_to_feature=False):
    mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
    feat_and_cont = extract_feature_and_contour_from_colour(
        mesh, append_contours_to_feature)
    mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                              requires_grad=False)
    face = torch.from_numpy(mesh.faces).t().to(torch.long).contiguous()
    mesh_colors = torch.tensor(mesh.visual.vertex_colors,
                               dtype=torch.float, requires_grad=False)
    data = Data(pos=mesh_verts, face=face, colors=mesh_colors,
                feat_and_cont=feat_and_cont)
    data = torch_geometric.transforms.FaceToEdge(False)(data)
    data.laplacian = torch.sparse_coo_tensor(
        *get_laplacian(data.edge_index, normalization='rw'))
    data = torch_geometric.transforms.GenerateMeshNormals()(data)
    return data


def extract_feature_and_contour_from_colour(colored,
                                            append_contour_to_feature=False):
    # assuming that the feature is colored in red and its contour in black
    if isinstance(colored, torch_geometric.data.Data):
        assert hasattr(colored, 'colors')
        colored_trimesh = torch_geometric.utils.to_trimesh(colored)
        colors = colored.colors.to(torch.long).numpy()
    elif isinstance(colored, trimesh.Trimesh):
        colored_trimesh = colored
        colors = colored_trimesh.visual.vertex_colors
    else:
        raise NotImplementedError

    graph = nx.from_edgelist(colored_trimesh.edges_unique)
    one_rings_indices = [list(graph[i].keys()) for i in range(len(colors))]

    features = {}
    for index, (v_col, i_ring) in enumerate(zip(colors, one_rings_indices)):
        if str(v_col) not in features:
            features[str(v_col)] = {'feature': [], 'contour': []}

        if is_contour(colors, index, i_ring):
            features[str(v_col)]['contour'].append(index)
        else:
            features[str(v_col)]['feature'].append(index)

    # certain vertices on the contour have interpolated colours ->
    # assign them to adjacent region
    elem_to_remove = []
    for key, feat in features.items():
        if len(feat['feature']) < 3:
            elem_to_remove.append(key)
            for idx in feat['feature']:
                counts = Counter([str(colors[ri])
                                  for ri in one_rings_indices[idx]])
                most_common = counts.most_common(1)[0][0]
                if most_common == key:
                    break
                features[most_common]['feature'].append(idx)
                features[most_common]['contour'].append(idx)
    for e in elem_to_remove:
        features.pop(e, None)

    # with b map
    # 0=eyes, 1=ears, 2=sides, 3=neck, 4=back, 5=mouth, 6=forehead,
    # 7=cheeks 8=cheekbones, 9=forehead, 10=jaw, 11=nose
    # key = list(features.keys())[11]
    # feature_idx = features[key]['feature']
    # contour_idx = features[key]['contour']

    # find surroundings
    # all_distances = self.compute_minimum_distances(
    #     colored.vertices, colored.vertices[contour_idx]
    # )
    # max_distance = max(all_distances)
    # all_distances[feature_idx] = max_distance
    # all_distances[contour_idx] = max_distance
    # threshold = 0.005
    # surrounding_idx = np.squeeze(np.argwhere(all_distances < threshold))
    # colored.visual.vertex_colors[surrounding_idx] = [0, 0, 0, 255]
    # colored.show()
    if append_contour_to_feature:
        for fc in features.values():
            fc['feature'] += fc['contour']
    return features


def is_contour(colors, center_index, ring_indices):
    center_color = colors[center_index]
    ring_colors = [colors[ri] for ri in ring_indices]
    for r in ring_colors:
        if not np.array_equal(center_color, r):
            return True
    return False


def compute_local_eigenvectors(template, k=50):
    evecs = {}
    for region_name, vertex_selection in template.feat_and_cont.items():
        edge_index_subset = subgraph(vertex_selection['feature'],
                                     template.edge_index,
                                     relabel_nodes=True)[0]
        graph_lapl = to_scipy_sparse_matrix(
            *get_laplacian(edge_index_subset, normalization=None))
        evecs[region_name] = eigsh(graph_lapl, k=k, which='SM')[1]
    return evecs


def to_torch_sparse(spmat):
    return torch.sparse_coo_tensor(
        torch.LongTensor([spmat.tocoo().row, spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def batch_mm(sparse, matrix_batch):
    """
    :param sparse: Sparse matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns (b, n, k) -> (n, b, k) -> (n, b*k)
    matrix = matrix_batch.transpose(0, 1).reshape(sparse.shape[1], -1)

    # And then reverse the reshaping.
    return sparse.mm(matrix).reshape(sparse.shape[0],
                                     batch_size, -1).transpose(1, 0)


def errors_to_colors(values, min_value=None, max_value=None, cmap=None):
    device = values.device
    min_value = values.min() if min_value is None else min_value
    max_value = values.max() if max_value is None else max_value
    if min_value != max_value:
        values = (values - min_value) / (max_value - min_value)

    cmapper = matplotlib.cm.get_cmap(cmap)
    values = cmapper(values.cpu().detach().numpy(), bytes=True)
    return torch.tensor(values[:, :, :3]).to(device)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(
                      os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def compute_signed_distances(x, template, std_x=None):
    if std_x is None:
        diff = x - template.pos
    else:
        diff = x * std_x  # if x is normalized equivalent to above
    signs = torch.sign((diff * template.norm).sum(dim=-1))
    modules = diff.norm(dim=-1, p=2)
    return (signs * modules).unsqueeze(-1)  # unsqueeze for mat multiplication


def annealing_coefficient(counter, total, percentage_of_total=100):
    return max(0, 1 - (100 * counter) / (percentage_of_total * total))


def plot_eigproj(eigp_mat, colors_as_str=None, out_path=None):
    eigproj_df = pd.DataFrame({"eig_" + str(idx): eigp_mat[:, idx].tolist()
                               for idx in range(eigp_mat.shape[1])})
    eigproj_df['gaussian'] = np.random.normal(0, 1, eigproj_df.shape[0])

    if colors_as_str is not None:
        colors = [np.fromstring(c[1:-1], sep=' ', dtype=int) for c in
                  colors_as_str]
        repeated_colors = [i for i in colors for _ in
                           range((eigproj_df.shape[1] - 1) // len(colors))]
        # repeated_colors = [np.zeros_like(repeated_colors[0])]+repeated_colors
        repeated_colors += [255 * np.ones_like(repeated_colors[0])]
        repeated_colors = [i / 255 for i in repeated_colors]
        my_cmap = ListedColormap(repeated_colors)
    else:
        my_cmap = None

    joypy.joyplot(eigproj_df, range_style='own', fade=True, ylabels=False,
                  overlap=2, colormap=my_cmap)

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
