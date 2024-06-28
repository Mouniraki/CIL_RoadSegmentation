import numpy as np
from collections import deque

from scipy.spatial.distance import cdist

WIDTH, HEIGHT = 25, 25



algos_and_params = {
    'mask_connected_though_border_radius':{
        'algorithm': 'mask_connected_though_border_radius',
        'radius': 3},
    'extend_path_to_closest':{
        'algorithm': 'extend_path_to_closest'},
    'connect_road':{
        'algorithm': 'connect_road',
        'max_dist': 25,
        'min_group_size': 8}
}



def coord_to_array(w, h):

    if w < 0 or h < 0 or h >= HEIGHT or w >= WIDTH:
        raise IndexError

    return w + h * 25


def mask_connected_though_border_radius(model_out_mask, params):
    radius = params['radius']
    mask_connected_though_border_radius_model_out_mask = model_out_mask.copy()
    assert WIDTH == HEIGHT
    queue_border = deque()
    for wh in range(HEIGHT):
        if mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, 0)] == 1:
            queue_border.append((wh, 0))
            mask_connected_though_border_radius_model_out_mask[
                coord_to_array(wh, 0)] = 2  # 2 = connected to border and discovered
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, HEIGHT-1)] == 1:
            queue_border.append((wh, HEIGHT-1))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, HEIGHT-1)] = 2
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(0, wh)] == 1:
            queue_border.append((0, wh))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(0, wh)] = 2
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(WIDTH-1, wh)] == 1:
            queue_border.append((WIDTH-1, wh))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(WIDTH-1, wh)] = 2
    while queue_border:
        w, h = queue_border.popleft()
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if 0 <= w + i < WIDTH and 0 <= h + j < HEIGHT and mask_connected_though_border_radius_model_out_mask[
                    coord_to_array(w + i, h + j)] == 1:
                    queue_border.append((w + i, h + j))
                    mask_connected_though_border_radius_model_out_mask[coord_to_array(w + i, h + j)] = 2
    return mask_connected_though_border_radius_model_out_mask == 2


def find_closest(mask_extended_road, w, h):
    assert WIDTH == HEIGHT

    sorted_radius_coord = sorted(zip(range(WIDTH // 2), range(HEIGHT // 2)),
        key=lambda coord: (coord[0] ** 2) + (coord[1] ** 2))[1:]

    for i, j in sorted_radius_coord:
        try:
            array_index = coord_to_array(i, j)
            if mask_extended_road[array_index] == 1:
                return (i, j)
        except IndexError:
            continue  # the index is out of the current frame no road can exist there
    return None

def homogenoeus_mix_list(list1, list2):
    if len(list1) == 0 : return list2
    if len(list2) == 0 : return list1
    if len(list1)<len(list2):
        list3 = list1
        list1 = list2
        list2 = list3

    interval = len(list1) // (len(list2))

    result = []
    for i in range(0, len(list1), interval):
        result.extend(list1[i:i + interval])
        if len(list2)>0:
            result.append(list2.pop(0))
    return result


def extend_road_between_points(mask_extended_road, w1, h1, w2, h2):
    nbr_step_horizontal = abs(w2-w1)
    nbr_step_up = abs(h2 - h1)
    step_direction_horizontal = 1 if w2 >= w1 else -1
    step_direction_up = 1 if h2 >= h1 else -1

    common_step = min(nbr_step_horizontal, nbr_step_up)

    common_steps = [(step_direction_horizontal, step_direction_up)] * common_step

    jump_nbr = abs(nbr_step_up - nbr_step_horizontal)
    jump_step = (0, 0)
    if nbr_step_up < nbr_step_horizontal:
        jump_step = (step_direction_horizontal, 0)
    elif nbr_step_up > nbr_step_horizontal:
        jump_step = (0, step_direction_up)
    jump_steps = [jump_step] * jump_nbr

    sequence_of_step = homogenoeus_mix_list(common_steps, jump_steps)

    coord_w, coord_h = w1, h1
    new_road_points = []
    for step in sequence_of_step:
        coord_w += step[0]
        coord_h += step[1]
        try:
            array_index = coord_to_array(coord_w, coord_h)
            mask_extended_road[array_index] = 1
            new_road_points.append((coord_w, coord_h))
        except IndexError:
            continue  # the index is out of the current frame no road can exist there
    assert coord_w == w2 and coord_h == h2
    return mask_extended_road, new_road_points




def merge_group(road_networks, w1, h1, w2, h2):
    if (w2, h2) not in road_networks:
        road_networks[(w2, h2)] = road_networks[(w1, h1)]
        return road_networks
    else:
        old_group_id = road_networks[(w1, h1)]
        new_group_id = road_networks[(w2, h2)]
        return {k: (new_group_id if v == old_group_id else v) for k, v in road_networks.items()}
'''
All patch around are part of a road 
Each road is a ground containing all patch part of it and its size
For each group find clothes group (pair of group distance -> pairwise distance between each element of both group take the smallest)
Remember where the shortest distance came from and call extend_road_between_points -> merge the two group 
Do this for all group 
If the shortest distance between two group is bigger than a threshold and the group is smaller than a threshold remove the small group
'''
def connect_road(model_out_mask, params):
    max_dist = params['max_dist']
    min_group_size = params['min_group_size']
    assert min_group_size >=1 and max_dist>=1
    mask_connect_road = model_out_mask.copy()
    road_networks = {} # (w,h) -> group_id

    # add all border as a group to which a road can be connected -> wihtout was giving good result
    for wh in range(0, WIDTH):
        road_networks[(-1, wh)] = 1
        road_networks[(WIDTH, wh)] = 1
        road_networks[(wh, -1)] = 1
        road_networks[(wh, HEIGHT)] = 1

    #find all subnetwork of road
    next_availible_id = 2
    for w in range(WIDTH):
        for h in range(HEIGHT):
            array_index = coord_to_array(w, h)
            if mask_connect_road[array_index] != 1:
                continue
            for coord in [(w+1, h), (w+1, h+1), (w, h+1)]:
                try:
                    array_index = coord_to_array(coord[0], coord[1])
                    if mask_connect_road[array_index] == 1:
                        if (w, h) not in road_networks:
                            road_networks[(w, h)] = next_availible_id
                            next_availible_id += 1
                        merge_group(road_networks, w, h, coord[0], coord[1])
                except IndexError:
                    continue  # the index is out of the current frame no road can exist there
    # Organize coordinates by their network ID
    subnetworks = [[] for _ in range(next_availible_id)]
    for coord, network_id in sorted(road_networks.items()):
        subnetworks[network_id-1].append(coord)

    subnetworks = list(filter(lambda net: len(net) >= min_group_size, subnetworks))


    # connect all subnetwork by computing for each pair of group the start and end of the closest connection between them
    subnetworks = sorted(subnetworks, key=len)
    while len(subnetworks) > 1:
        group_to_merge = subnetworks[0]
        current_min = np.inf
        min_group2 = None
        min_group1_coord = (None, None)
        min_group2_coord = (None, None)

        for group_id in range(1, len(subnetworks)):
            target_group = subnetworks[group_id]
            distances = cdist(group_to_merge, target_group)
            min_index = np.argmin(distances)
            i, j = np.unravel_index(min_index, distances.shape)
            if distances[i, j] < current_min:
                current_min = distances[i, j]
                min_group2 = group_id
                min_group1_coord = group_to_merge[i]
                min_group2_coord = target_group[j]

        if current_min < max_dist:
            mask_connect_road, new_road_points = extend_road_between_points(mask_connect_road, min_group1_coord[0], min_group1_coord[1], min_group2_coord[0], min_group2_coord[1])
            subnetworks[min_group2].extend(new_road_points)
            subnetworks[min_group2].extend(group_to_merge)
            subnetworks = subnetworks[1:]
        else:
            # we simply remove the smallest group and consider it as a wrong prediction
            if len(group_to_merge) < len(subnetworks[min_group2]):
                subnetworks = subnetworks[1:]
            else:
                subnetworks = subnetworks[:min_group2]+subnetworks[min_group2+1:]

    return mask_connect_road





def extend_path_to_closest(model_out_mask, params):
    mask_extended_road = model_out_mask.copy()
    assert WIDTH == HEIGHT
    for w in range(WIDTH):
        for h in range(HEIGHT):
            try:
                if mask_extended_road[coord_to_array(w, h)] == 1:
                    clossest_road = find_closest(mask_extended_road, w, h)
                    if clossest_road is not None:
                        mask_extended_road = extend_road_between_points(mask_extended_road, w, h, clossest_road[0],
                                                                        clossest_road[1])
            except IndexError:
                continue

    return mask_extended_road


def patch_postprocessing(patch_postprocessing_label, algo_and_params=None):
    patch_postprocessing_label_return = []
    nbr_img = patch_postprocessing_label.size // (25 * 25)
    for i_img in range(nbr_img):
        if algo_and_params['algorithm'] == 'mask_connected_though_border_radius':
            patch_postprocessing_label_return = np.append(patch_postprocessing_label_return,
                                                          mask_connected_though_border_radius(
                                                              patch_postprocessing_label[
                                                              i_img * 25 * 25:(i_img + 1) * 25 * 25], algo_and_params))
        elif algo_and_params['algorithm'] == 'extend_path_to_closest':
            raise NotImplementedError
            patch_postprocessing_label_return = np.append(patch_postprocessing_label_return,
                                                          extend_path_to_closest(patch_postprocessing_label[
                                                                                 i_img * 25 * 25:(i_img + 1) * 25 * 25], algo_and_params))
        elif algo_and_params['algorithm'] == "connect_road":
            patch_postprocessing_label_return = np.append(patch_postprocessing_label_return,
                                                          connect_road(patch_postprocessing_label[
                                                                                 i_img * 25 * 25:(i_img + 1) * 25 * 25], algo_and_params))

    return patch_postprocessing_label_return
    fig, axs = plt.subplots(h_patches, w_patches, figsize=figsize)
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(np.maximum(p, np.array([l.item(), 0., 0.])))
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()
