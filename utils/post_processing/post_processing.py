import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MaxPool2d

class PostProcessing:
    def __init__(self, device, postprocessing_patch_size=16):
        self.postprocessing_patch_size = postprocessing_patch_size
        self.device = device

    '''
    Private method refer to connect_roads for documentation
    '''

    def _connect_road(self, 
                      mask_labeled_roads, 
                      max_dist, 
                      min_group_size, 
                      fat):
        # this method get a image where each pixel has a group id from which he is part
        unique_labels = torch.unique(mask_labeled_roads, sorted=False,
                                     return_inverse=False)  # we get the list of group's labels
        unique_labels = unique_labels[unique_labels != 0]  # 0 represent nothing in our array,
        road_coordinates = []
        for unique_label in unique_labels:
            road_coordinate = (
                        mask_labeled_roads == unique_label).nonzero()  # we extract per group the list of coordinates being part of it
            road_coordinates.append(road_coordinate)

        # connect all subnetwork by computing for each pair of group the start and end of the closest connection between them
        #subnetworks = sorted(road_coordinates, key=len)
        subnetworks = list(filter(lambda subnetwork: len(subnetwork) > min_group_size, road_coordinates))

        while len(subnetworks) > 1:
            current_min = np.inf  # the smallest distance between two groups
            min_group_id1, min_group_id2 = None, None  # the group having the smallest gap
            min_group_coord1, min_group_coord2 = (None, None), (None, None)  # the exact position of the samallest gap
            for group_id1 in range(len(subnetworks)):
                for group_id2 in range(group_id1 + 1, len(subnetworks)):
                    group1, group2 = subnetworks[group_id1].float(), subnetworks[group_id2].float()
                    distances = torch.cdist(group1, group2,
                                            p=2.0)  # pairwise distances between every points of the group pair
                    min_index = torch.argmin(distances)
                    i, j = torch.unravel_index(min_index, distances.shape)  # the coordinate index in the two group
                    if distances[
                        i, j] < current_min:  # if the group is closer than precedent other group pair we save the position
                        current_min = distances[i, j]
                        min_group_id1, min_group_id2 = group_id1, group_id2
                        min_group_coord1 = group1[i]
                        min_group_coord2 = group2[j]
            if current_min <= max_dist:  # group are within max distance threshold from each other and thus a path can be created between them else this postprocessing method bascially consider the pair already together (merge the group without creating a path between them
                new_road_points = self.__cordinates_between_points(min_group_coord1, min_group_coord2, fat=fat).to(
                    self.device)  # create a path of width fat between the two closest's group's points
                subnetworks[min_group_id1] = torch.cat([subnetworks[min_group_id1], new_road_points])
            subnetworks[min_group_id1] = torch.cat([subnetworks[min_group_id1], subnetworks[
                min_group_id2]])  # consider the two closest group as one big unity
            subnetworks.pop(min_group_id2)

        #convert coordinates to mask
        mask_connect_road = torch.zeros(mask_labeled_roads.shape).to(self.device)
        for subnetwork in subnetworks:
            for point in subnetwork.long():
                if point[0] < 0 or point[0] >= mask_connect_road.shape[0] or point[1] < 0 or point[1] >= \
                        mask_connect_road.shape[1]: continue
                mask_connect_road[point[0], point[1]] = 1

        return mask_connect_road

    '''
    Following the principle of Bresenham's line algorithm
    '''

    def __cordinates_between_points(self, 
                                    p1, 
                                    p2, 
                                    fat=0):
        # added principe of kernel for each newly added point we add all point around it in a window of size 'fat'
        if fat == 0:
            fat_kernel = torch.tensor([[0, 0]]).to(self.device)
        else:
            fat_kernel = torch.tensor([[i, j] for i in range(-fat, fat) for j in range(-fat, fat)]).to(self.device)

        x1, y1 = p1[0].item(), p1[1].item()
        x2, y2 = p2[0].item(), p2[1].item()

        result_coordinate = torch.empty((0, 2)).to(self.device)

        dx, dy = abs(x2 - x1), abs(y2 - y1)
        step_x = 1 if x1 < x2 else -1
        step_y = 1 if y1 < y2 else -1

        error = dx - dy

        # walk in direction of the second point while saving every pixel's coordinates we walk on
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
                break  # avoid writing the end coordinate

            #result_coordinate = torch.cat([result_coordinate, torch.tensor([[x1, y1]])])
            result_coordinate = torch.cat([result_coordinate, fat_kernel + torch.tensor([x1, y1]).to(self.device)])
            result_coordinate = torch.unique(result_coordinate, sorted=False, return_inverse=False, return_counts=False,
                                             dim=0)
        return result_coordinate.to(self.device)

    '''
    All patch around are part of a road 
    Each road is a ground containing all pixel touching in it
    For each group find clothes group (pair of group distance -> pairwise distance between each element of both group take the smallest)
    For the two group with the shortest distance between them, if this distance is smaller than max_dist then connect the two group by adding the shortest possible road to connect the two 
    Do this for all group 
    Remove group that are too small
    @param downsample : specify the diminished resolution at which the method works 
    @param max_dist : specify the maximum distance between the two groups (border touching - isolated to be considered touching)
    @param min_group_size : minimum size of a road prediction that is not filtered out as being considered a wrong prediction
    @param threshold_road_not_road : value at which we consider the confidence of the model to be prediciting a road
    '''

    def connect_roads(self, 
                      mask_connect_roads, 
                      downsample=2, 
                      max_dist=25, 
                      min_group_size=1, 
                      threshold_road_not_road=0,
                      fat=2):
        assert min_group_size >= 1 and max_dist >= 1 and downsample >= 1
        # downsample the predicted mask
        negative_confidence, positive_confidence = mask_connect_roads.min().item(), mask_connect_roads.max().item()
        m = nn.AvgPool2d(downsample, stride=downsample)
        mask_connect_roads = m(mask_connect_roads) >= threshold_road_not_road # create binary mask by using a cutoff (road - not road)
        batch_size = mask_connect_roads.shape[0]
        height = mask_connect_roads.shape[2]
        width = mask_connect_roads.shape[3]

        # add an infinite road as the border
        mask_connect_roads_padded = F.pad(mask_connect_roads, (1, 1, 1, 1), mode='constant', value=0)
        biggest_label = (height + 2) * (width + 2) + 1
        # give each road pixel a label
        labels = torch.arange(1, (height + 2) * (width + 2) + 1, dtype=torch.int32).reshape(height + 2,
                                                                                            width + 2).repeat(
            batch_size, 1, 1, 1).to(self.device) * mask_connect_roads_padded

        #iterate until all touching pixel of road have the same value -> propagate the smalll label to all pixel touching (end with all pixel being part of the same group having the same value
        while True:
            labels_with_road = (labels + ((~mask_connect_roads_padded) * (
                        biggest_label + 10))).float()  # the part where model predicted no road are thrown to infinity such that the min pulling doesn't merge a road group with it
            m = nn.MaxPool2d(3, stride=1,
                             padding=1)  # take neighbouring pixel, doesn't reduce spatial dimension, per default negative infinity on the side
            new_labels = -m(-labels_with_road)
            new_labels *= mask_connect_roads_padded  # not road stay not road ignore road label we previously gave
            if torch.equal(labels, new_labels):
                break
            labels = new_labels

        # process each mask of the batch separately (GPU memory constraints)
        result = torch.empty((0, 1, height, width)).to(self.device)
        for i in range(batch_size):
            connected_mask = self._connect_road(labels[i, 0, :, :], max_dist, min_group_size, fat).unsqueeze(
                0).unsqueeze(0)
            connected_mask = connected_mask[:, :, 1:-1, 1:-1]  # remove the border infinite road padding
            connected_mask = (connected_mask != 0) * positive_confidence + (
                        connected_mask == 0) * negative_confidence  # reorder as the model 0 -> negative, 1 -> positiv
            result = torch.cat([result, connected_mask], dim=0)
        m = nn.Upsample(scale_factor=downsample, mode='nearest')
        result = m(result)
        return result

    '''
    This method filter out of the mask every pixel that are not though other pixel connected to a border of the image 
    @param downsample : specify the diminished resolution at which the method works 
    @param contact_radius : specify the maximum distance between the two groups (border touching - isolated to be considered touching)
    @param threshold_road_not_road : value at which we consider the confidence of the model to be prediciting a road
    '''

    def mask_connected_though_border_radius(self,
                                            mask_connect_roads, 
                                            downsample=2, 
                                            contact_radius=3,
                                            threshold_road_not_road=0):
        assert contact_radius >= 1 and contact_radius % 2 == 1 and downsample >= 1
        negative_confidence, positive_confidence = mask_connect_roads.min().item(), mask_connect_roads.max().item()
        # downsample the predicted mask
        m = nn.AvgPool2d(downsample, stride=downsample)
        mask_connect_roads = (m(mask_connect_roads) >= threshold_road_not_road).float()

        # add a specific value representing the border
        labels = F.pad(mask_connect_roads, (1, 1, 1, 1), mode='constant',
                       value=10.)  # outside border are given value 10
        # add an infinite road as the border
        mask_connect_roads_padded = F.pad(mask_connect_roads, (1, 1, 1, 1), mode='constant', value=1)

        # iterate until all touching pixel of road have the same value (propagate the border specific label though touching pixels)
        while True:
            m = nn.MaxPool2d(contact_radius, stride=1, padding=((
                                                                            contact_radius - 1) // 2))  # take neighbouring pixel, doesn't reduce spatial dimension, per default negative infinity on the side
            new_labels = m(labels)  # propagate the border value which is bigger than the rest  to road within radius
            new_labels *= mask_connect_roads_padded  # not road stay not road
            if torch.equal(labels, new_labels):
                break
            labels = new_labels

        mask_connect_roads_padded = (labels == 10)  # keep only road which exit the image (connected to a border)
        mask_connect_roads_padded = (mask_connect_roads_padded != 0) * positive_confidence + (
                    mask_connect_roads_padded == 0) * negative_confidence  # reorder as the model 0 -> negative, 1 -> positiv (take maximum confidence after postprocessing)
        mask_connect_roads_padded = mask_connect_roads_padded[:, :, 1:-1, 1:-1]  # remove the oustide border padding we added
        m = nn.Upsample(scale_factor=downsample, mode='nearest')
        mask_connect_roads_padded = m(mask_connect_roads_padded)  # reverse downsampling

        return mask_connect_roads_padded

    def blurring_averaging(self, 
                           mask_connect_roads, 
                           kernel_size=7):
        assert kernel_size % 2 == 1
        m = nn.AvgPool2d(kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        mask_connect_roads_blurred = m(mask_connect_roads)
        return mask_connect_roads_blurred

    '''
    (Warning this method is very slow! -> She wasn't showing real improvement to the model performance so we kept it like this.)
    The goal of this function is to remove every gaps between close pixel by connecting every pixels distanciated by less than distance_max directly by the shortest path to each other
    @param downsample : specify the diminished resolution at which the method works 
    @param distance_max : distance below one to pixel should be connected by there shortest path to each other
    @param threshold_road_not_road : value at which we consider the confidence of the model to be prediciting a road
    '''

    def connect_all_close_pixels(self, 
                                 mask_connect_roads, 
                                 downsample=2, 
                                 distance_max=10, 
                                 threshold_road_not_road=0):
        assert distance_max >= 1 and downsample >= 1
        # downsample the predicted mask
        m = nn.AvgPool2d(downsample, stride=downsample)
        mask_connect_roads = (m(mask_connect_roads) >= threshold_road_not_road).float()
        batch_size = mask_connect_roads.shape[0]
        height = mask_connect_roads.shape[2]
        width = mask_connect_roads.shape[3]

        result = torch.zeros((0, 1, height, width)).to(self.device)

        # process batch sample individually (GPU memory constraints)
        for b in range(batch_size):
            mask_connect_road = mask_connect_roads[b, 0, :, :]
            road_coordinates = mask_connect_road.nonzero(
                as_tuple=False).float()  # extract coordinate of all road pixels

            acc_result = torch.empty((0, 2)).to(self.device)
            acc_result = torch.cat([acc_result, road_coordinates], dim=0)
            # for each pixel predicted as part of a road
            for road_coordinate in road_coordinates:
                candidate_road_coordinates = road_coordinates[
                    (road_coordinates[:, 0] <= road_coordinate[0] + distance_max) * (
                                road_coordinates[:, 0] >= road_coordinate[0]) * (
                                road_coordinates[:, 1] >= road_coordinate[1]) * (
                                road_coordinates[:, 1] <= road_coordinate[
                            1] + distance_max)]  #look only down, right -> up, left connection are made from upper, left road pixels
                for candidate_road_coordinate in candidate_road_coordinates: # check the distance between all pixels in neighbouring window and create a path between them if the distance is smaller than a threshold
                    dist = torch.norm(road_coordinate - candidate_road_coordinate)
                    if dist > distance_max or dist <= 1: continue  #alraedy neighbooring pixel are ignored no road needed to connect them
                    added_coordinates = self.__cordinates_between_points(road_coordinate, candidate_road_coordinate)
                    acc_result = torch.cat([acc_result, added_coordinates], dim=0)
            acc_result = torch.unique(acc_result, sorted=False, return_inverse=False, return_counts=False, dim=0) # keep only unique pixels

            mask_connect_road = torch.zeros(mask_connect_road.shape).to(self.device)
            for point in acc_result.long(): # convert list of road pixels back to mask
                mask_connect_road[point[0], point[1]] = 1

            result = torch.cat([result, mask_connect_road.unsqueeze(0).unsqueeze(0)], dim=0) # reassemble the batch

        m = nn.Upsample(scale_factor=downsample, mode='nearest')
        result = m(result)  # reverse downsampling

        return result
