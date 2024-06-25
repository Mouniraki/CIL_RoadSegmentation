
import numpy as np
from collections import deque
def coord_to_array(w, h):
    return w + h * 25


def mask_connected_though_border_radius(model_out_mask, radius=3):
    mask_connected_though_border_radius_model_out_mask = model_out_mask.copy()
    width, height = 25, 25
    assert width == height
    queue_border = deque()
    for wh in range(height):
        if mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, 0)] == 1:
            queue_border.append((wh, 0))
            mask_connected_though_border_radius_model_out_mask[
                coord_to_array(wh, 0)] = 2  # 2 = connected to border and discovered
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, -1)] == 1:
            queue_border.append((wh, -1))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, -1)] = 2
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(0, wh)] == 1:
            queue_border.append((0, wh))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(0, wh)] = 2
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(-1, wh)] == 1:
            queue_border.append((-1, wh))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(-1, wh)] = 2
    while queue_border:
        w, h = queue_border.popleft()
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if 0 <= w + i < width and 0 <= h + j < height and mask_connected_though_border_radius_model_out_mask[
                    coord_to_array(w + i, h + j)] == 1:
                    queue_border.append((w + i, h + j))
                    mask_connected_though_border_radius_model_out_mask[coord_to_array(w + i, h + j)] = 2
    return mask_connected_though_border_radius_model_out_mask == 2


def patch_postprocessing(patch_postprocessing_label, radius=3):
    patch_postprocessing_label_return = []
    nbr_img = patch_postprocessing_label.size // (25 * 25)
    for i_img in range(nbr_img):
        patch_postprocessing_label_return = np.append(patch_postprocessing_label_return,
                                                      mask_connected_though_border_radius(patch_postprocessing_label[
                                                                                          i_img * 25 * 25:(
                                                                                                                      i_img + 1) * 25 * 25],
                                                                                          radius=radius))
    return patch_postprocessing_label_return
    fig, axs = plt.subplots(h_patches, w_patches, figsize=figsize)
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(np.maximum(p, np.array([l.item(), 0., 0.])))
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()