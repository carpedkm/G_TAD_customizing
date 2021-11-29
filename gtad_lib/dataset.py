# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.testing._private.utils import break_cycles
import pandas as pd
import json, pickle
import torch.utils.data as data
import torch
import h5py
import math

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors) # 겹치는게 존재하기만 하면 됨
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):  # thumos
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"]  # 256
        self.temporal_gap = 1. / self.temporal_scale # 1/256
        self.subset = subset
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.feat_dim = opt['feat_dim']
        # Assuming someone wont outsmart this by mutating the dict 😝.
        # Consider to use YACS and FB code structure in the future.
        self.cfg = opt
        if self.subset == 'train':
            self.boundary_file = './temporal_info.json'
        else :
            self.boundary_file = './temporal_info_test.json'
        self._temporal_file_load()

        #### THUMOS
        self.skip_videoframes = opt['skip_videoframes']
        self.num_videoframes = opt['temporal_scale'] # how does this work ?
        self.max_duration = opt['max_duration']
        self.min_duration = opt['min_duration']
        if self.feature_path[-3:]=='200':
            self.feature_dirs = [self.feature_path + "/flow/csv", self.feature_path + "/rgb/csv"]
        else:
            self.feature_dirs = [self.feature_path]
        self._get_data()
        self.video_list = self.data['video_names']
        # self._getDatasetDict()
        self._get_match_map()

    def _getDatasetDict(self): # so, it's not used then
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def _get_video_data(self, data, index):
        return data['video_data'][index]

# check
    def __getitem__(self, index):
        video_data = self._get_video_data(self.data, index) # get one from 2793 // get data stored from _get_data one by one through index
        video_data = torch.tensor(video_data.transpose()) # by transposing, make it feature_dim by length (window size)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index) # get the label data : gt bbox for that matching windows, anchor_xmins, anchor_xmaxs : (corresponding anchors that's multiplied by 5 to compare with gt in frames)
            return video_data, confidence_score, match_score_start, match_score_end # confidence score : 64 by 256
        else:
            return index, video_data # when inferencing -> just input the video_data (2048 by 256)

    def _get_match_map(self):
        match_map = []
        for idx in range(self.num_videoframes): # 256 (temporal_scale in opts.py)
            tmp_match_window = []
            xmin = self.temporal_gap * idx # 1/256 * idx ??
            for jdx in range(1, self.max_duration + 1): # For THUMOS 14, it's 1 to 64 (max_duration is 64)
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])  # [0,0.01], [0,0.02], ... 64 x 2
            match_map.append(tmp_match_window)  # 256 x 64 x 2
        match_map = np.array(match_map)  # 256 x 64 x 2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100], 64 x 256 x 2
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # (duration x start) x 2
        self.match_map = match_map  # duration is same in row, start is same in col
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.temporal_scale + 1)]
    
    # Methods customized
    def _temporal_file_load(self):
        self.temporal_set = load_json(self.boundary_file)

        
    def _update_transition_sites(self, top_transition_sites, per_interval_count_):
        for cnt, reg in enumerate(top_transition_sites):
            if cnt == 0:
                prev_reg = reg
                continue
            else:
                diff = reg - prev_reg
                if diff <= per_interval_count_ * 4: # 만약 특정 구간에서 인터벌이 20(0, 20)인 상태에서 6개를 샘플링해야 하는 상황일 때, 20을 top_transition_sites에서 삭제하게 된다.
                        top_transition_sites.remove(reg)
                        break
                else :
                    prev_reg = reg
        return top_transition_sites
            
    def _update_intervals(self, num_video_frames_, top_transition_sites_):
        intervals_count_= len(top_transition_sites_) - 1
        per_interval_count_ = math.floor(num_video_frames_ / intervals_count_)
        per_interval_count_list_ = [per_interval_count_ for _ in range(intervals_count_)]
        complementary_count_ = num_video_frames_ - (per_interval_count_ * intervals_count_)
        return intervals_count_, per_interval_count_, complementary_count_, per_interval_count_list_
    
    def _length_calculate(self, top_transition_list):
        tmp = []
        for cnt, i in enumerate(top_transition_list):
            if cnt == 0:
                end = i
            else :
                start = end
                end = i
                interval = end - start
                tmp.append(interval)
        return tmp
    
    def _get_train_label(self, index):
        # change the measurement from second to percentage
        # gt_bbox = []
        gt_iou_map = []
        gt_bbox = self.data['gt_bbox'][index]
        anchor_xmin = self.data['anchor_xmins'][index] # from _get_data
        anchor_xmax = self.data['anchor_xmaxs'][index] # from _get_data
        offset = int(min(anchor_xmin))
        for j in range(len(gt_bbox)):
            # tmp_info = video_labels[j]
            tmp_start = max(min(1, (gt_bbox[j][0]-offset)*self.temporal_gap/self.skip_videoframes), 0) # to use it with the given feature data, it's kind of shifting and scaling
            tmp_end =   max(min(1, (gt_bbox[j][1]-offset)*self.temporal_gap/self.skip_videoframes), 0)
            if tmp_end > 1:
                print('tmp end larger than 1', tmp_end)
            # gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.max_duration,self.num_videoframes])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        # if not gt_iou_map.max()>0.9:
        #     raise ValueError
        gt_iou_map = torch.Tensor(gt_iou_map)

        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.skip_videoframes
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        # calculate the ioa for all timestamp - here, use the original frame counts // not the percentages
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1]))) # rerturns 각각의 GT에 대해 
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)

    def _get_data(self):
        if 'train' in self.subset:
            anno_df = pd.read_csv(self.video_info_path+'val_Annotation.csv')
        elif 'val' in self.subset:
            anno_df = pd.read_csv(self.video_info_path+'test_Annotation.csv')

        video_name_list = sorted(list(set(anno_df.video.values[:])))

        video_info_dir = '/'.join(self.video_info_path.split('/')[:-1])
        saved_data_path = os.path.join(video_info_dir, 'saved.%s.%s.nf%d.sf%d.num%d.%s.pkl' % (
            self.feat_dim, self.subset, self.num_videoframes, self.skip_videoframes,
            len(video_name_list), self.mode)
                                       )
        print(saved_data_path)
        if not self.cfg['override'] and os.path.exists(saved_data_path):
            print('Got saved data.')
            with open(saved_data_path, 'rb') as f:
                self.data, self.durations = pickle.load(f)
            print('Size of data: ', len(self.data['video_names']), flush=True)
            return

        if self.feature_path:
            list_data = []

        list_anchor_xmins = []
        list_anchor_xmaxs = []
        list_gt_bbox = []
        list_videos = []
        list_indices = []

        num_videoframes = self.num_videoframes # the window size is 256
        skip_videoframes = self.skip_videoframes # so it's 5. -> is it the sigma? for the averaging?
        start_snippet = int((skip_videoframes + 1) / 2) 
        stride = int(num_videoframes / 2) # so the stride is 128 for the thumos14

        self.durations = {}

        self.flow_val = h5py.File(self.feature_path+'/flow_val.h5', 'r')
        self.rgb_val = h5py.File(self.feature_path+'/rgb_val.h5', 'r')
        self.flow_test = h5py.File(self.feature_path+'/flow_test.h5', 'r')
        self.rgb_test = h5py.File(self.feature_path+'/rgb_test.h5', 'r')

        for num_video, video_name in enumerate(video_name_list):
            print('Getting video %d / %d' % (num_video, len(video_name_list)), flush=True)
            anno_df_video = anno_df[anno_df.video == video_name] # for the each video sequence (same video) -> they are grouped as one set, in df
            temporal_set = self.temporal_set[video_name]
            
            top_transition_sites = temporal_set['top_transition_sites']
            total_frames = temporal_set['frames']
            
            if self.mode == 'train':
                gt_xmins = anno_df_video.startFrame.values[:] # Get the start frame values from pandas df
                gt_xmaxs = anno_df_video.endFrame.values[:] # Get the end frame values from pandas df

    # UPDATED CODE -------------------------------------------------------
            intervals_count = len(top_transition_sites) - 1
            per_interval_count = math.floor(num_videoframes / intervals_count) # per interval, there should be sampled in this amount
            complementary_count = num_videoframes - (per_interval_count * intervals_count) # use this amount to add up the sampling count to the top "complementary_count" numbers
            
            if len(top_transition_sites) == 2: # for really short video input, with the similarity matrix that shows no transition sites
                flag = 1
            else: # otherwise, we should calculate it as,
                flag = 0
                
                while flag != 1: # update the transition sites -> for ex) if there exists 6 interval count for the interval length 20, then you shouldn't pick it up, rather delete the transition -> update
                    old_top = top_transition_sites # 기존 transition site list
                    top_transition_sites = self._update_transition_sites(top_transition_sites, per_interval_count) # update the transition site (delete the non-necessary element)
                    intervals_count, per_interval_count, complementary_count, per_interval_count_list = self._update_intervals(num_videoframes, top_transition_sites) # recalculte the intervals count, per_interval_count, complementary_count
                    if old_top == top_transition_sites: # If there's no more changes
                        flag = 1 # set the flag 1, and stop it
                
                diff_in_transitions = self._length_calculate(top_transition_sites) # calculate the length of each intervals
                sorted_transitions = sorted(diff_in_transitions, reverse=True) # sort it
                idx_list_transitions = []
                
                for i in diff_in_transitions:
                    idx_list_transitions.append(sorted_transitions.index(i)) # put the index inside
                
                for cnt, i in enumerate(idx_list_transitions): # since it's not 256 yet, add the complementary to the k(comp count) largest interval values
                    if i < complementary_count:
                        per_interval_count_list[cnt] += 1
            
                
            # Integrate and average the feature -----
                # select the frames to sample -> use the transition sites 
                # -> and pick it by using the updated per_interval_count_list
                sampling_frames_list = []
                for cnt, i in enumerate(top_transition_sites):
                    if cnt == 0:
                        end = i
                    else:
                        start = end
                        end = i
                        int_length = end - start# interval length
                        k = 0
                        sampling_rate = float(int_length) / float(per_interval_count_list[cnt - 1])
                        while sampling_rate * k < int_length:
                            sampling_frames_list.append(math.floor(sampling_rate * k + start))
                            k += 1
                        
            # Select the rows from the sampling_frames_list
                # load in the features
            if 'val' in video_name:
                feature_h5s = [
                    self.flow_val[video_name],
                    self.rgb_val[video_name]
                ]
            elif 'test' in video_name:
                feature_h5s = [
                    self.flow_test[video_name],
                    self.rgb_test[video_name]
                ]
            
         
            # select the rows and then average it -> stack it up
            if len(top_transition_sites) == 2: # impossible to stack up and average it -> pick up the features with the desired amount (by linear interpolation)
                # use linear interpolation to fix this case
                gap = float(top_transition_sites[1] - top_transition_sites[0]) / float(num_videoframes)
                for num in range(num_videoframes):
                    if num == 0:
                        flow_stack = feature_h5s[0][0, :]
                        rgb_stack = feature_h5s[1][0, :]
                    else :
                        loc = gap * num
                        floor_ = math.floor(loc)
                        ceil_ = math.ceil(loc)
                        lambda_floor = loc - floor_
                        lambda_ceil = ceil_ - loc # equiv to 1 - lambda_floor
                        flow_stack = np.vstack((flow_stack, feature_h5s[0][floor_, :] * lambda_floor + feature_h5s[0][ceil_, :] * lambda_ceil))
                        rgb_stack = np.vstack((rgb_stack, feature_h5s[1][floor_, :] * lambda_floor + feature_h5s[1][ceil_, :] * lambda_ceil))
                    
            else:
                for i in range(total_frames):
                    if i == 0:
                        end = i
                    elif i == sampling_frames_list[1]:
                        start = end
                        end = i
                        flow_stack = np.average(feature_h5s[0][start:end, :], axis=0)
                        rgb_stack  = np.average(feature_h5s[1][start:end, :], axis=0)
                    elif i in sampling_frames_list:
                        start = end
                        end = i
                        flow_stack = np.vstack((flow_stack, np.average(feature_h5s[0][start:end, :], axis=0)))
                        rgb_stack = np.vstack((rgb_stack, np.average(feature_h5s[1][start:end, :], axis=0)))
                    elif i == total_frames - 1:
                        start = end
                        end = total_frames - 1
                        flow_stack = np.vstack((flow_stack, np.average(feature_h5s[0][start:end, :], axis=0)))
                        rgb_stack = np.vstack((rgb_stack, np.average(feature_h5s[1][start:end, :], axis=0)))
                    else :
                        pass
                
                # make it as concatenated feature_h5s like original code
                feature_h5s = [
                    flow_stack,
                    rgb_stack
                ]
    # update END -----------------------------------------
            num_snippet = min([h5.shape[0] for h5 in feature_h5s]) # so, 1018 is the num_snippet ? looked like it was shape[1] the snippet from paper
            
            df_data = np.concatenate([h5[:num_snippet, :] # Yes, it's the number of snippets, and as we can see in the paper, 1018 * 5 is the original frame amounts
                                      for h5 in feature_h5s], # It's because, the number is given with downsampled TSN encoded.
                                    axis=1) # concatenate, for example, 1018 by 1024 two of them -> 1018 by 2048, so the feature dim is concatenated.

            # df_snippet = [start_snippet + skip_videoframes * i for i in range(num_snippet)] 
            df_snippet = [skip_videoframes * i for i in range(num_snippet)] # skip video frames how does it work?
            num_windows = int((num_snippet + stride - num_videoframes) / stride) # num_videoframes = 256 (window size), stride = 128 (half of window size)
            windows_start = [i * stride for i in range(num_windows)] # windows_start : the start frame x -> the row index, you should think
            if num_snippet < num_videoframes: # so if the number of snippet (?) is smaller than the window size? -> why do they call this num_snippet?
                windows_start = [0]
                # Add on a bunch of zero data if there aren't enough windows.
                tmp_data = np.zeros((num_videoframes - num_snippet, self.feat_dim))
                df_data = np.concatenate((df_data, tmp_data), axis=0)
                df_snippet.extend([
                    df_snippet[-1] + skip_videoframes * (i + 1)
                    for i in range(num_videoframes - num_snippet)
                ])
            elif num_snippet - windows_start[-1] - num_videoframes > int(num_videoframes / skip_videoframes):
                windows_start.append(num_snippet - num_videoframes)

            for start in windows_start:
                tmp_data = df_data[start:start + num_videoframes, :] # so this stores the data

                tmp_snippets = np.array(df_snippet[start:start + num_videoframes]) # why this? -> in order to use the window -> 256 frames // df_snippet contains : 0, 5, 10, ... , => meaning : retrieving the original frame index, not the downsampled one?
                if self.mode == 'train':
                    tmp_anchor_xmins = tmp_snippets - skip_videoframes / 2.
                    tmp_anchor_xmaxs = tmp_snippets + skip_videoframes / 2.
                    tmp_gt_bbox = []
                    tmp_ioa_list = []
                    for idx in range(len(gt_xmins)): # gt_xmins , gt_xmaxs : use original frame (not the 5 divided ones): in order to compare, tmp_anchor -> they should be consisted of original frames also. 
                        tmp_ioa = ioa_with_anchors(gt_xmins[idx], gt_xmaxs[idx],
                                                   tmp_anchor_xmins[0],
                                                   tmp_anchor_xmaxs[-1]) # e.g. xmins, xmaxs of GT -> compare with the minimum of tmp anchor and maximum of tmp anchor 
                                                                         #-> get IoU score over anchor 
                                                                         #-> return the score
                        tmp_ioa_list.append(tmp_ioa) # check if the window contains the given ground truth ts and te
                        if tmp_ioa > 0: # if there's contained ones -> append all of them to tmp_gt_bbox
                            tmp_gt_bbox.append([gt_xmins[idx], gt_xmaxs[idx]]) # meaning, that in this window, this GT is inside

                    if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > 0.9: # in that video seq, if there's case that includes the gt for the window, and if there exists ioa larger than 0.9
                        list_gt_bbox.append(tmp_gt_bbox) # add that tmp_gt_bbox to list_gt_bbox (adding that gt set that includes ioa larger than 0.9 to list_gt_bbox) 
                        list_anchor_xmins.append(tmp_anchor_xmins) # in that case, add the tmp_anchor_xmins
                        list_anchor_xmaxs.append(tmp_anchor_xmaxs) # in that case, add the tmp_anchor_xmaxs
                        list_videos.append(video_name) # in that case, add the video_name (like video_validation_0000051)
                        list_indices.append(tmp_snippets) # tmp snippet inside that window (256), that's re-stored to the original frame index, multiplied by 5
                        if self.feature_dirs:
                            list_data.append(np.array(tmp_data).astype(np.float32))
                elif "infer" in self.mode:
                    list_videos.append(video_name)
                    list_indices.append(tmp_snippets)
                    list_data.append(np.array(tmp_data).astype(np.float32)) # tmp_data has the video feature that corresponds to the windows, list_data contains 

        print("List of videos: ", len(set(list_videos)), flush=True)
        self.data = {
            'video_names': list_videos,
            'indices': list_indices # tmp_snippets
        }
        if self.mode == 'train':
            self.data.update({
                'gt_bbox': list_gt_bbox,
                'anchor_xmins': list_anchor_xmins,
                'anchor_xmaxs': list_anchor_xmaxs,
            })
        if self.feature_dirs:
            self.data['video_data'] = list_data  # list data stores 256 (window size) by 2048 features -> a lot of them
        print('Size of data: ', len(self.data['video_names']), flush=True)
        with open(saved_data_path, 'wb') as f:
            pickle.dump([self.data, self.durations], f)
        print('Dumped data...')


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a, b, c, d in train_loader:
        print(a.shape,b.shape,c.shape,d.shape)
        break
