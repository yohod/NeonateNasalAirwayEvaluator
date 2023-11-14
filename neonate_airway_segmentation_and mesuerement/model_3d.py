import os
import trimesh
import numpy as np
from skimage.measure import marching_cubes
import cc3d
# from skimage.filters import _gaussian



def reconstruction3d(seg_images, xy_spacing, thickness, connected=True, case="", save_path =""):
    """
    Reconstruct a 3D model from segmented images using Marching Cubes algorithm. save the model

    :param seg_images: List of segmented images.
    :param xy_spacing: Tuple representing the pixel spacing in the (coronal, sagittal) directions.
    :param thickness: Thickness of the axial slices.
    :param connected: Boolean indicating whether to use connected components before reconstruction. Default is True.
    :param case: Case number or identifier for saving the 3D model. If not provided, the user will be prompted to enter it.

    :return: None,
    """
    if connected:
        images = cc3d.largest_k(seg_images, 1, connectivity=18)
        images[images == 1] = 255
    else:
        images = seg_images

    # Padding with zeros to receive a closed volume
    black_image = np.zeros_like(images[0])
    images = [black_image] + list(images) + [black_image]

    # Find the verts and faces for the 3D images using Marching Cubes algorithm
    spacing = thickness, xy_spacing[0], xy_spacing[1]
    verts, faces = marching_cubes(np.array(images), level=None, spacing=spacing, allow_degenerate=False,
                                  method='lewiner')[:2]  # degenerate = False?

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # need to smooth it
    # https://docs.scipy.org/doc/scipy/tutorial/interpolate.html
    # https: // plotly.com / python / smoothing /
    # https://trimsh.org/trimesh.smoothing.html

    # smooth_mesh = trimesh.smoothing.filter_humphrey(mesh)# alpha=0.1, beta=0.7, iterations=100)
    # smooth_mesh = trimesh.smoothing.filter_laplacian(mesh)
    # smooth_mesh = trimesh.smoothing.filter_mut_dif_laplacian(mesh)
    # smooth_mesh = trimesh.smoothing.filter_taubin(mesh)


    # Save the 3D model
    os.chdir(save_path)
    mesh.export("model " + case + ".stl")





