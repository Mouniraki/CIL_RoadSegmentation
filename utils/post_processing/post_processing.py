import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MaxPool2d

WIDTH, HEIGHT = 25, 25
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class PostProcessing:
    ''''
    postprocessing_patch_size : represent the downscale factor
    '''

    def __init__(self, postprocessing_patch_size=16):
        self.postprocessing_patch_size = postprocessing_patch_size

    def filter_isolated_patches(self, pred, max_distance):
        height = pred.shape[2]
        width = pred.shape[3]
        raise NotImplementedError

    def __label_connected_components(self, image):
        assert image.ndim == 2, "Image should be a 2D tensor"
        assert torch.all((image == 0.0) | (image == 1.0)), "Image should be binary with values 0 and 1"

        labels = labels[1:-1, 1:-1]
        unique_labels, new_labels = torch.unique(labels, return_inverse=True)
        labeled_image = new_labels.reshape(labels.shape)
        return labeled_image

    # idea downscale
    # filter isolated
    # connect all together
    # blur
    # threshold

    def connect_road_segements(self, pred, downsample=8, max_dist=25, min_group_size=1):
        assert min_group_size >= 1 and max_dist >= 1
        if downsample <= 8:
            print("We recommand using a downsample factor relatively big to reduce runtime")
        pred = pred.copy()

        pred = pred > 0.5
        padded_image = F.pad(pred, (1, 1, 1, 1), mode='constant', value=0)
        height, width = padded_image.shape
        labels = torch.arange(1, height * width + 1, dtype=torch.int32).reshape(height, width) * padded_image
        kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

        while True:
            labels_updated = -MaxPool2d(-labels, 3, 1).squeeze(0)
            if torch.equal(labels, labels_updated):
                break
            labels = labels_updated
        # extract the group of road patch

    def _connect_road(self, mask_labeled_roads, max_dist, min_group_size):

        unique_labels = torch.unique(mask_labeled_roads, sorted=False, return_inverse=False) # 0 represent nothing in our array
        unique_labels = unique_labels[unique_labels != 0]
        road_coordinates = []
        for unique_label in unique_labels:
            road_coordinate = (mask_labeled_roads == unique_label).nonzero()
            road_coordinates.append(road_coordinate)





        # connect all subnetwork by computing for each pair of group the start and end of the closest connection between them
        #subnetworks = sorted(road_coordinates, key=len)
        subnetworks = list(filter(lambda subnetwork: len(subnetwork) > min_group_size, road_coordinates))

        while len(subnetworks) > 1:
            current_min = np.inf
            min_group_id1, min_group_id2 = None, None
            min_group_coord1, min_group_coord2 = (None, None), (None, None)
            for group_id1 in range(len(subnetworks)):
                for group_id2 in range(group_id1+1, len(subnetworks)):
                    group1, group2 = subnetworks[group_id1].float(), subnetworks[group_id2].float()
                    distances = torch.cdist(group1, group2, p=2.0)
                    min_index = torch.argmin(distances)
                    i, j = torch.unravel_index(min_index, distances.shape)
                    if distances[i, j] < current_min:
                        current_min = distances[i, j]
                        min_group_id1, min_group_id2 = group_id1, group_id2
                        min_group_coord1 = group1[i]
                        min_group_coord2 = group2[j]
            if current_min <= max_dist: # group are within max distance threshold from each other and can thus a path can be created between them else this postprocessing method bascially consider the pair already together
                new_road_points = self.__cordinates_between_points(min_group_coord1, min_group_coord2).to(DEVICE)
                subnetworks[min_group_id1] = torch.cat([subnetworks[min_group_id1], new_road_points])
            subnetworks[min_group_id1] = torch.cat([subnetworks[min_group_id1], subnetworks[min_group_id2]])
            subnetworks.pop(min_group_id2)


        #convert coordinates to mask
        mask_connect_road = torch.zeros(mask_labeled_roads.shape).to(DEVICE)
        for subnetwork in subnetworks:
            for point in subnetwork.long():
                mask_connect_road[point[0], point[1]] = 1

        return mask_connect_road

    '''
    Following the principle of Bresenham's line algorithm'''
    def __cordinates_between_points(self, p1, p2):
        x1, y1 = p1[0].item(), p1[1].item()
        x2, y2 = p2[0].item(), p2[1].item()

        result_coordinate = torch.empty((0, 2))

        dx, dy = abs(x2 - x1), abs(y2 - y1)
        step_x = 1 if x1 < x2 else -1
        step_y = 1 if y1 < y2 else -1

        error = dx - dy

        while True:
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x1 += step_x
            if e2 < dx:
                error += dx
                y1 += step_y

            if x1 == x2 and y1 == y2:
                break # avoid writing the end coordinate

            result_coordinate = torch.cat([result_coordinate, torch.tensor([[x1, y1]])])
        return result_coordinate

    '''
    All patch around are part of a road 
    Each road is a ground containing all patch part of it and its size
    For each group find clothes group (pair of group distance -> pairwise distance between each element of both group take the smallest)
    Remember where the shortest distance came from and call extend_road_between_points -> merge the two group 
    Do this for all group 
    If the shortest distance between two group is bigger than a threshold and the group is smaller than a threshold remove the small group
    '''

    def connect_roads(self, mask_connect_road, downsample=2, max_dist=25, min_group_size=1, threshold_road_not_road=0):
        assert min_group_size >= 1 and max_dist >= 1
        negative_confidence = mask_connect_road.min().item()
        positive_confidence = mask_connect_road.max().item()
        m = nn.AvgPool2d(downsample, stride=downsample)
        mask_connect_road = m(mask_connect_road) >= threshold_road_not_road
        batch_size = mask_connect_road.shape[0]
        height = mask_connect_road.shape[2]
        width = mask_connect_road.shape[3]

        mask_connect_road_padded = F.pad(mask_connect_road, (1, 1, 1, 1), mode='constant', value=0)
        biggest_label = (height + 2) * (width + 2) + 1
        labels = torch.arange(1, (height + 2) * (width + 2) + 1, dtype=torch.int32).reshape(height + 2,
                                                                                            width + 2).repeat(
            batch_size, 1, 1, 1).to(DEVICE) * mask_connect_road_padded

        #iterate until all touching pixel of road have the same value
        while True:
            labels_with_road = (labels + ((~mask_connect_road_padded) * (biggest_label+10))).float() # the part where model predicted no road are thrown to ifinity such that the min pulling doesn't merge a road group with it
            m = nn.MaxPool2d(3, stride=1, padding=1) # take neighbouring pixel, doesn't reduce spatial dimension, per default negative infinity on the side
            new_labels = -m(-labels_with_road)
            new_labels *= mask_connect_road_padded # not road stay not road ignore road label we previously gave
            if torch.equal(labels, new_labels):
                break
            labels = new_labels


        #labels = labels[:,:, 1:-1, 1:-1] #remove virtual border road


        result = torch.empty((0, 1, height, width)).to(DEVICE)
        for i in range(batch_size):
            connected_mask = self._connect_road(labels[i, 0, :, :], max_dist, min_group_size).unsqueeze(0).unsqueeze(0)
            connected_mask = connected_mask[:,:, 1:-1, 1:-1]# remove the border infinite road padding
            connected_mask = (connected_mask != 0)*positive_confidence + (connected_mask == 0)*negative_confidence# reorder as the model 0 -> negative, 1 -> positiv
            result = torch.cat([result, connected_mask], dim=0)
        m = nn.Upsample(scale_factor=downsample, mode='nearest')
        result = m(result)
        return result


algos_and_params = {
    'mask_connected_though_border_radius': {
        'algorithm': 'mask_connected_though_border_radius',
        'radius': 3},
    'extend_path_to_closest': {
        'algorithm': 'extend_path_to_closest'},
    'connect_road': {
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
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, HEIGHT - 1)] == 1:
            queue_border.append((wh, HEIGHT - 1))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(wh, HEIGHT - 1)] = 2
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(0, wh)] == 1:
            queue_border.append((0, wh))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(0, wh)] = 2
        elif mask_connected_though_border_radius_model_out_mask[coord_to_array(WIDTH - 1, wh)] == 1:
            queue_border.append((WIDTH - 1, wh))
            mask_connected_though_border_radius_model_out_mask[coord_to_array(WIDTH - 1, wh)] = 2
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
    if len(list1) == 0: return list2
    if len(list2) == 0: return list1
    if len(list1) < len(list2):
        list3 = list1
        list1 = list2
        list2 = list3

    interval = len(list1) // (len(list2))

    result = []
    for i in range(0, len(list1), interval):
        result.extend(list1[i:i + interval])
        if len(list2) > 0:
            result.append(list2.pop(0))
    return result


def extend_road_between_points(mask_extended_road, w1, h1, w2, h2):
    nbr_step_horizontal = abs(w2 - w1)
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
    assert min_group_size >= 1 and max_dist >= 1
    mask_connect_road = model_out_mask.copy()
    road_networks = {}  # (w,h) -> group_id

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
            for coord in [(w + 1, h), (w + 1, h + 1), (w, h + 1)]:
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
        subnetworks[network_id - 1].append(coord)

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
            mask_connect_road, new_road_points = extend_road_between_points(mask_connect_road, min_group1_coord[0],
                                                                            min_group1_coord[1], min_group2_coord[0],
                                                                            min_group2_coord[1])
            subnetworks[min_group2].extend(new_road_points)
            subnetworks[min_group2].extend(group_to_merge)
            subnetworks = subnetworks[1:]
        else:
            # we simply remove the smallest group and consider it as a wrong prediction
            if len(group_to_merge) < len(subnetworks[min_group2]):
                subnetworks = subnetworks[1:]
            else:
                subnetworks = subnetworks[:min_group2] + subnetworks[min_group2 + 1:]

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
                                                                                 i_img * 25 * 25:(i_img + 1) * 25 * 25],
                                                                                 algo_and_params))
        elif algo_and_params['algorithm'] == "connect_road":
            patch_postprocessing_label_return = np.append(patch_postprocessing_label_return,
                                                          connect_road(patch_postprocessing_label[
                                                                       i_img * 25 * 25:(i_img + 1) * 25 * 25],
                                                                       algo_and_params))

    return patch_postprocessing_label_return
    fig, axs = plt.subplots(h_patches, w_patches, figsize=figsize)
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(np.maximum(p, np.array([l.item(), 0., 0.])))
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()
