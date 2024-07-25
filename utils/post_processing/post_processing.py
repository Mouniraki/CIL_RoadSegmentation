import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MaxPool2d
import torchvision
import torchvision.transforms as T

WIDTH, HEIGHT = 25, 25
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class PostProcessing:


    def __init__(self, postprocessing_patch_size=16):
        self.postprocessing_patch_size = postprocessing_patch_size


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
    def connect_roads(self, mask_connect_roads, downsample=2, max_dist=25, min_group_size=1, threshold_road_not_road=0):
        assert min_group_size >= 1 and max_dist >= 1
        negative_confidence, positive_confidence = mask_connect_roads.min().item(), mask_connect_roads.max().item()
        m = nn.AvgPool2d(downsample, stride=downsample)
        mask_connect_roads = m(mask_connect_roads) >= threshold_road_not_road
        batch_size = mask_connect_roads.shape[0]
        height = mask_connect_roads.shape[2]
        width = mask_connect_roads.shape[3]

        mask_connect_roads_padded = F.pad(mask_connect_roads, (1, 1, 1, 1), mode='constant', value=0)
        biggest_label = (height + 2) * (width + 2) + 1
        labels = torch.arange(1, (height + 2) * (width + 2) + 1, dtype=torch.int32).reshape(height + 2,
                                                                                            width + 2).repeat(
            batch_size, 1, 1, 1).to(DEVICE) * mask_connect_roads_padded

        #iterate until all touching pixel of road have the same value
        while True:
            labels_with_road = (labels + ((~mask_connect_roads_padded) * (biggest_label+10))).float() # the part where model predicted no road are thrown to ifinity such that the min pulling doesn't merge a road group with it
            m = nn.MaxPool2d(3, stride=1, padding=1) # take neighbouring pixel, doesn't reduce spatial dimension, per default negative infinity on the side
            new_labels = -m(-labels_with_road)
            new_labels *= mask_connect_roads_padded # not road stay not road ignore road label we previously gave
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


    '''
    This method filter out of the mask every pixel that are not though other pixel connected to a border of the image 
    @param downsample : specify the diminished resolution at which the method works 
    @param contact_radius : specify the maximum distance between the two groups (border touching - isolated to be considered touching)
    @param threshold_road_not_road : value at which we consider the confidence of the model to be prediciting a road
    '''
    def mask_connected_though_border_radius(self, mask_connect_roads, downsample=2, contact_radius=3, threshold_road_not_road=0):
        assert contact_radius >= 1 and contact_radius % 2 == 1
        negative_confidence, positive_confidence = mask_connect_roads.min().item(), mask_connect_roads.max().item()
        m = nn.AvgPool2d(downsample, stride=downsample)
        mask_connect_roads = (m(mask_connect_roads) >= threshold_road_not_road).float()
        batch_size = mask_connect_roads.shape[0]
        height = mask_connect_roads.shape[2]
        width = mask_connect_roads.shape[3]

        labels = F.pad(mask_connect_roads, (1, 1, 1, 1), mode='constant', value=10.) # outside border are given value 10
        mask_connect_roads_padded = F.pad(mask_connect_roads, (1, 1, 1, 1), mode='constant', value=1)

        # iterate until all touching pixel of road have the same value
        while True:
            m = nn.MaxPool2d(contact_radius, stride=1, padding=((contact_radius-1)//2))  # take neighbouring pixel, doesn't reduce spatial dimension, per default negative infinity on the side
            new_labels = m(labels) # propagate the border value which is bigger than the rest  to road within radius
            new_labels *= mask_connect_roads_padded  # not road stay not road
            if torch.equal(labels, new_labels):
                break
            labels = new_labels

        mask_connect_roads_padded = (labels==10) # keep only road which exit the image (connected to a border)
        mask_connect_roads_padded = (mask_connect_roads_padded != 0)*positive_confidence + (mask_connect_roads_padded == 0)*negative_confidence # reorder as the model 0 -> negative, 1 -> positiv (take maximum confidence after postprocessing
        mask_connect_roads_padded = mask_connect_roads_padded[:,:,1:-1,1:-1] # remove the oustide border padding we added
        m = nn.Upsample(scale_factor=downsample, mode='nearest')
        mask_connect_roads_padded = m(mask_connect_roads_padded) # reverse downsampling

        return mask_connect_roads_padded

    def blurring_and_threshold(self, mask_connect_roads, kernel_size=7):
        assert kernel_size % 2 == 1
        m = nn.AvgPool2d(kernel_size, stride=1, padding=(kernel_size-1)//2)
        mask_connect_roads_blurred = m(mask_connect_roads)
        return mask_connect_roads_blurred


















