import os
from copy import deepcopy
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import random
import pandas as pd
import soundfile as sf
from random import sample
from torch.utils.data import DataLoader
import torchaudio

torchaudio.set_audio_backend("sox_io")


class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)
        return img


def extract_audio_features(audio_path, fps, n_frames):
    audio, sr = sf.read(audio_path)

    if audio.ndim == 2:
        audio = audio.mean(-1)

    frame_n_samples = int(sr / fps)
    curr_length = len(audio)

    target_length = frame_n_samples * n_frames

    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])

    shifted_n_samples = 0
    curr_feats = []

    for i in range(n_frames):
        curr_samples = audio[i * frame_n_samples:shifted_n_samples + i * frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1),
                                                     sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1)  # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)

    return curr_feats


def custom_collate(batch):
    listener_emotion_clip_personal = [item[0] for item in batch if item[0].shape[0] > 0]
    listener_3dmm_clip_personal = [item[1] for item in batch if item[1].shape[0] > 0]
    listeners_label_personal = [item[2] for item in batch if item[2].shape[0] > 0]

    length = len(listener_emotion_clip_personal)
    if length > 0:
        listener_emotion_clip_personal = torch.cat(listener_emotion_clip_personal, dim=0)
        listener_3dmm_clip_personal = torch.cat(listener_3dmm_clip_personal, dim=0)
        listeners_label_personal = torch.cat(listeners_label_personal, dim=0)
    else:
        listener_emotion_clip_personal = torch.zeros(size=(0,))
        listener_3dmm_clip_personal = torch.zeros(size=(0,))
        listeners_label_personal = torch.zeros(size=(0,))

    return (
        listener_emotion_clip_personal,
        listener_3dmm_clip_personal,
        listeners_label_personal,
    )


class ReactionDataset(data.Dataset):
    def __init__(self, root_path, split, method="speaker_based", img_size=256, crop_size=224, num_person=16,
                 num_sample=4, clip_length=751, fps=25, load_emotion_l=False, load_3dmm_l=False):

        self._root_path = root_path
        self._num_person = num_person
        self._num_sample = num_sample
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self._method = method
        self._data_path = os.path.join(self._root_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)

        neighbour_emotion_path = os.path.join(root_path, 'neighbour_emotion_' + split + '.npy')
        self.neighbour_emotion = np.load(neighbour_emotion_path)

        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_l = load_emotion_l

        self.dataset_path = os.path.join(root_path, self._split)
        self._audio_path = os.path.join(self.dataset_path, 'Audio_files')
        self._video_path = os.path.join(self.dataset_path, 'Video_files')
        self._emotion_path = os.path.join(self.dataset_path, 'Emotion')
        self._3dmm_path = os.path.join(self.dataset_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)

        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = [path for path in list(self._list_path.values[:, 1])]
        listener_path = [path for path in list(self._list_path.values[:, 2])]

        speaker_path_tmp = speaker_path + listener_path
        listener_path_tmp = listener_path + speaker_path
        speaker_path = speaker_path_tmp
        listener_path = listener_path_tmp

        self.speaker_path = speaker_path.copy()
        self.listener_path = listener_path.copy()
        self.listener_ids = ['/'.join(path.split('/')[:3]) for path in listener_path]

        listener_ids_set = []
        for e in self.listener_ids:
            if e not in listener_ids_set:
                listener_ids_set.append(e)
        label_mapping = {e: i for i, e in enumerate(listener_ids_set)}
        # the labels for all listeners
        self.listener_labels = [label_mapping[listener_id] for listener_id in self.listener_ids]

        self.listener_ids_dict = {}
        for i, listener_id in enumerate(self.listener_ids):
            if listener_id not in self.listener_ids_dict:
                self.listener_ids_dict[listener_id] = [i]
            else:
                self.listener_ids_dict[listener_id].append(i)

        self.data_list = [path for path in list(self._list_path.values[:, 1])] + [path for path in list(
            self._list_path.values[:, 2])]

        self._len = len(self.data_list)  # 3186

    def __getitem__(self, index):

        # =================== Find Similar Speakers & Appropriate Reactions ===================
        speaker_line = self.neighbour_emotion[index]  # self.neighbour_emotion.shape: (3186, 3186)
        sim_speakers_index = np.where(speaker_line == True)[0]

        relative_indices_personal = []

        if self._method == "non_speaker_based":
            listener_ids_dict = deepcopy(self.listener_ids_dict)
            listener_names = list(listener_ids_dict.keys())
            sampled_listeners = sample(listener_names, self._num_person)

            for name in sampled_listeners:
                if len(listener_ids_dict[name]) > self._num_sample:
                    selected_indices = np.random.permutation(len(listener_ids_dict[name]))[:self._num_sample]
                    relative_indices = [listener_ids_dict[name][i] for i in selected_indices]
                    relative_indices_personal.extend(relative_indices)

        elif self._method == "speaker_based":
            appro_listeners_index = sim_speakers_index
            appro_listeners_ids = [self.listener_ids[idx] for idx in appro_listeners_index]
            appro_listeners = {}
            for i in range(len(appro_listeners_index)):
                if appro_listeners_ids[i] not in appro_listeners:
                    appro_listeners[appro_listeners_ids[i]] = [appro_listeners_index[i]]
                else:
                    appro_listeners[appro_listeners_ids[i]].append(appro_listeners_index[i])

            temp_list = []
            for _, v in appro_listeners.items():
                if 1 < len(v) <= 16:
                    temp_list.append(v)
                elif len(v) > 16:
                    temp_list.append([v[i] for i in np.random.permutation(len(v))[:16]])
            if len(temp_list) > 4:
                temp_list = \
                    [temp_list[j] for j in np.random.permutation(len(temp_list))[:4]]

            for e in temp_list:
                relative_indices_personal.extend(e)
        else:
            raise ValueError("Personal-specific mathod {} is not implemented.".format(self._method))

        listeners_label_personal = torch.asarray([self.listener_labels[idx] for idx in relative_indices_personal])

        total_length = 751
        cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0

        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion_clip_personal = torch.zeros(size=(0,))
        if self.load_emotion_l:
            selected_listener_emotion = []
            union_index = relative_indices_personal

            for i in union_index:
                listener_emotion_path = os.path.join(self._emotion_path, self.listener_path[i] + '.csv')

                if 'NoXI' in listener_emotion_path:
                    listener_emotion_path = listener_emotion_path.replace('Novice_video', 'P2')
                    listener_emotion_path = listener_emotion_path.replace('Expert_video', 'P1')

                if 'Emotion/RECOLA/group' in listener_emotion_path:
                    listener_emotion_path = listener_emotion_path.replace('P25', 'P1')
                    listener_emotion_path = listener_emotion_path.replace('P26', 'P2')
                    listener_emotion_path = listener_emotion_path.replace('P41', 'P1')
                    listener_emotion_path = listener_emotion_path.replace('P42', 'P2')
                    listener_emotion_path = listener_emotion_path.replace('P45', 'P1')
                    listener_emotion_path = listener_emotion_path.replace('P46', 'P2')

                listener_emotion_path = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
                emotion = torch.from_numpy(np.array(listener_emotion_path.drop(0)).astype(np.float32))[
                          cp: cp + self._clip_length]

                selected_listener_emotion.append(emotion)

            # shape: [..., _clip_length, 25]
            if len(selected_listener_emotion):
                listener_emotion_clip_personal = torch.stack(selected_listener_emotion)

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm_clip_personal = torch.zeros(size=(0,))
        # ====== load listener 3dmm
        if self.load_3dmm_l:
            selected_listener_3dmm = []
            union_index = relative_indices_personal

            for i in union_index:
                listener_3dmm_path = os.path.join(self._3dmm_path, self.listener_path[i] + '.npy')
                listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
                listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
                listener_3dmm = self._transform_3dmm(listener_3dmm)[0]
                selected_listener_3dmm.append(listener_3dmm)

            # shape: [..., _clip_length, 25]
            if len(selected_listener_3dmm):
                listener_3dmm_clip_personal = torch.stack(selected_listener_3dmm)

        return (
            listener_emotion_clip_personal,
            listener_3dmm_clip_personal,
            listeners_label_personal,
        )

    def __len__(self):
        return self._len


def get_dataloader(conf):
    assert conf.split in ["train", "val", "test"], "split must be in [train, val, test]"
    print('==> Preparing data for {}...'.format(conf.split) + '\n')

    dataset = ReactionDataset(root_path=conf.dataset_path,
                              split=conf.split,
                              method=conf.method,
                              num_person=conf.num_person,
                              num_sample=conf.num_sample,
                              img_size=conf.img_size,
                              crop_size=conf.crop_size,
                              clip_length=conf.clip_length,
                              fps=25,
                              load_emotion_l=conf.load_emotion_l,
                              load_3dmm_l=conf.load_3dmm_l)

    dataloader = DataLoader(dataset=dataset, collate_fn=custom_collate, batch_size=conf.batch_size,
                            shuffle=conf.shuffle,
                            num_workers=conf.num_workers)

    return dataloader
