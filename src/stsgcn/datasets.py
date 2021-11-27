import os
import torch
import numpy as np
import pickle
from stsgcn.data_utils import ang2joint, readCSVasFloat, find_indices_srnn, expmap2xyz_torch, find_indices_256


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_n, output_n, skip_rate, body_model_dir, actions=None, split=0):
        pass

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        item = self.p3d[key][fs]
        return item


class Amass_3D_Dataset(Dataset):
    def __init__(self, data_dir, input_n, output_n, skip_rate, body_model_dir, actions=None, split=0):
        """
        adapted from
        https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/amass3d.py

        :param data_dir: path_to_data
        :param actions: always None
        :param input_n: number of input frames
        :param output_n: number of output frames
        :param split: 0 train, 1 validation, 2 test
        """
        self.path_to_data = os.path.join(data_dir, 'amass')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)  # start from 4 for 17 joints, removing the non moving ones
        self.skip_rate = skip_rate
        seq_len = self.in_n + self.out_n

        amass_splits = [
            ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
            ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
            ['BioMotionLab_NTroje']]
        # amass_splits = [['ACCAD'],
        #                 ['HumanEva'],
        #                 ['BioMotionLab_NTroje']]

        skel = np.load(body_model_dir)  # load mean skeleton
        p3d0 = torch.from_numpy(skel['p3d0']).float()
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]
        n = 0
        for ds in amass_splits[split]:
            if not os.path.isdir(os.path.join(self.path_to_data, ds)):
                print(ds)
                continue
            print('Loading {}...'.format(ds))
            for sub in os.listdir(os.path.join(self.path_to_data, ds)):
                if not os.path.isdir(os.path.join(self.path_to_data, ds, sub)):
                    continue
                for act in os.listdir(os.path.join(self.path_to_data, ds, sub)):
                    if not act.endswith('.npz'):
                        continue
                    pose_all = np.load(os.path.join(self.path_to_data, ds, sub, act))
                    try:
                        poses = pose_all['poses']
                    except:
                        print('no poses at {}_{}_{}'.format(ds, sub, act))
                        continue
                    frame_rate = pose_all['mocap_framerate']
                    fn = poses.shape[0]
                    sample_rate = int(frame_rate // 25)
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float()
                    poses = poses.reshape([fn, -1, 3])
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint(p3d0_tmp, poses, parent)
                    self.p3d.append(p3d.data.numpy())
                    if split == 2:
                        valid_frames = np.arange(0, fn - seq_len + 1, self.skip_rate)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, self.skip_rate)

                    self.keys.append((ds, sub, act))
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1


class H36M_Ang_Dataset(Dataset):
    def __init__(self, data_dir, input_n, output_n, skip_rate, body_model_dir, actions=None, split=0):
        """
        adapted from
        https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion.py

        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 test
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'h3.6m/dataset')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.seq = {}
        self.data_idx = []

        self.dimensions_to_use = np.array(
            [6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
             43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86])
        self.dimensions_to_ignore = np.array(
            [[0, 1, 2, 3, 4, 5, 10, 11, 16, 17, 18, 19, 20, 25, 26, 31, 32, 33, 34, 35, 48, 49, 50, 58,
              59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
              98]])

        seq_len = self.in_n + self.out_n
        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions

        subs = subs[split]

        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        # the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        # remove global rotation and translation
                        the_sequence[:, 0:6] = 0
                        # p3d = data_utils.expmap2xyz_torch(the_sequence)
                        self.seq[(subj, action, subact)] = the_sequence

                        valid_frames = np.arange(0, num_frames - seq_len + 1)

                        tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                else:
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    # the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_sequence1[:, 0:6] = 0
                    # p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    self.seq[(subj, action, 1)] = the_sequence1

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    # the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_sequence2[:, 0:6] = 0
                    # p3d2 = data_utils.expmap2xyz_torch(the_seq2)
                    self.seq[(subj, action, 2)] = the_sequence2

                    # fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                    #                                                 input_n=self.in_n)
                    fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len,
                                                         input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [(subj, action, 1)] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [(subj, action, 2)] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))


class H36M_3D_Dataset(Dataset):
    def __init__(self, data_dir, input_n, output_n, skip_rate, body_model_dir, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 test
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'h3.6m/dataset')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
            # acts = ["walking", "eating"]
        else:
            acts = actions
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float()
                        # remove global rotation and translation
                        the_sequence[:, 0:6] = 0
                        p3d = expmap2xyz_torch(the_sequence)
                        self.p3d[key] = p3d.view(num_frames, -1).data.numpy()

                        valid_frames = np.arange(0, num_frames - seq_len + 1)

                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1
                else:
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float()
                    the_seq1[:, 0:6] = 0
                    p3d1 = expmap2xyz_torch(the_seq1)
                    self.p3d[key] = p3d1.view(num_frames1, -1).data.numpy()

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float()
                    the_seq2[:, 0:6] = 0
                    p3d2 = expmap2xyz_torch(the_seq2)

                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).data.numpy()

                    fs_sel1, fs_sel2 = find_indices_256(num_frames1, num_frames2, seq_len,
                                                        input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 2

        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)


class DPW_3D_Dataset(Dataset):
    def __init__(self, data_dir, input_n, output_n, skip_rate, body_model_dir, split=0):
        """
        Adapted from
        https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/dpw3d.py
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 test
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, '3dpw/sequenceFiles')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        self.skip_rate = skip_rate
        seq_len = self.in_n + self.out_n

        if split == 0:
            data_path = self.path_to_data + '/train/'
        elif split == 1:
            data_path = self.path_to_data + '/validation/'
        elif split == 2:
            data_path = self.path_to_data + '/test/'
        files = []
        for (dirpath, dirnames, filenames) in os.walk(data_path):
            files.extend(filenames)

        skel = np.load(body_model_dir)
        p3d0 = torch.from_numpy(skel['p3d0']).float()[:, :22]
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            parent[i] = parents[i]
        n = 0

        sample_rate = int(60 // 25)

        for f in files:
            with open(data_path + f, 'rb') as f:
                print('>>> loading {}'.format(f))
                data = pickle.load(f, encoding='latin1')
                joint_pos = data['poses_60Hz']
                for i in range(len(joint_pos)):
                    poses = joint_pos[i]
                    fn = poses.shape[0]
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float()
                    poses = poses.reshape([fn, -1, 3])
                    poses = poses[:, :-2]
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
                    # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                    self.p3d.append(p3d.data.numpy())

                    if split == 2:
                        # valid_frames = np.arange(0, fn - seq_len + 1, opt.skip_rate_test)
                        # valid_frames = np.arange(0, fn - seq_len + 1, 2)
                        valid_frames = np.arange(0, fn - seq_len + 1)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, self.skip_rate)

                    # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1
