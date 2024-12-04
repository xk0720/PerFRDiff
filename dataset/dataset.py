import os
from copy import deepcopy
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import random
import pandas as pd
from PIL import Image
import soundfile as sf
import av
from decord import VideoReader
from decord import cpu
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


def custom_collate_train(batch):
    speaker_audio_clip = [item[0] for item in batch if item[0].shape[0] > 0]
    speaker_video_clip = [item[1] for item in batch if item[1].shape[0] > 0]
    speaker_emotion_clip = [item[2] for item in batch if item[2].shape[0] > 0]
    speaker_3dmm_clip = [item[3] for item in batch if item[3].shape[0] > 0]
    listener_video_clip = [item[4] for item in batch if item[4].shape[0] > 0]
    # listener_video_clip_personal = [item[5] for item in batch if item[5].shape[0] > 0]
    listener_emotion_clip = [item[6] for item in batch if item[6].shape[0] > 0]
    # listener_emotion_clip_personal = [item[7] for item in batch if item[7].shape[0] > 0]
    listener_3dmm_clip = [item[8] for item in batch if item[8].shape[0] > 0]
    listener_3dmm_clip_personal = [item[9] for item in batch if item[9].shape[0] > 0]
    listener_reference = [item[10] for item in batch if item[10].shape[0] > 0]

    if len(speaker_audio_clip) > 0:
        speaker_audio_clip = torch.stack(speaker_audio_clip, dim=0)
    else:
        speaker_audio_clip = torch.zeros(size=(0,))

    if len(speaker_video_clip) > 0:
        speaker_video_clip = torch.stack(speaker_video_clip, dim=0)
    else:
        speaker_video_clip = torch.zeros(size=(0,))

    speaker_emotion_clip = torch.stack(speaker_emotion_clip, dim=0)
    speaker_3dmm_clip = torch.stack(speaker_3dmm_clip, dim=0)

    if len(listener_video_clip) > 0:
        listener_video_clip = torch.stack(listener_video_clip, dim=0)
    else:
        listener_video_clip = torch.zeros(size=(0,))

    listener_emotion_clip = torch.stack(listener_emotion_clip, dim=0)
    _, _, l, emotion_dim = listener_emotion_clip.shape
    listener_emotion_clip = listener_emotion_clip.reshape(-1, l, emotion_dim)

    listener_3dmm_clip = torch.stack(listener_3dmm_clip, dim=0)
    _, _, l, _3dmm_dim = listener_3dmm_clip.shape
    listener_3dmm_clip = listener_3dmm_clip.reshape(-1, l, _3dmm_dim)
    listener_3dmm_clip_personal = torch.stack(listener_3dmm_clip_personal, dim=0)
    l = listener_3dmm_clip_personal.shape[-2]
    listener_3dmm_clip_personal = listener_3dmm_clip_personal.reshape(-1, l, _3dmm_dim)

    if len(listener_reference) > 0:
        listener_reference = torch.stack(listener_reference, dim=0)
    else:
        listener_reference = torch.zeros(size=(0,))

    return (
        speaker_audio_clip,
        speaker_video_clip,
        speaker_emotion_clip,
        speaker_3dmm_clip,
        listener_video_clip,
        # listener_video_clip_personal,
        listener_emotion_clip,
        # listener_emotion_clip_personal,
        listener_3dmm_clip,
        listener_3dmm_clip_personal,
        listener_reference,
    )


def custom_collate_test(batch):
    speaker_audio_clip = [item[0] for item in batch if item[0].shape[0] > 0]
    speaker_video_clip = [item[1] for item in batch if item[1].shape[0] > 0]
    speaker_emotion_clip = [item[2] for item in batch if item[2].shape[0] > 0]
    speaker_3dmm_clip = [item[3] for item in batch if item[3].shape[0] > 0]
    listener_video_clip = [item[4] for item in batch if item[4].shape[0] > 0]
    listener_emotion_clip = [item[5] for item in batch if item[5].shape[0] > 0]
    # listener_emotion_clip_personal = [item[6] for item in batch if item[6].shape[0] > 0]
    listener_3dmm_clip = [item[7] for item in batch if item[7].shape[0] > 0]
    listener_3dmm_clip_personal = [item[8] for item in batch if item[8].shape[0] > 0]
    listener_reference = [item[9] for item in batch if item[9].shape[0] > 0]

    speaker_audio_clip = torch.stack(speaker_audio_clip, dim=0)

    if len(speaker_video_clip) > 0:
        speaker_video_clip = torch.stack(speaker_video_clip, dim=0)
    else:
        speaker_video_clip = torch.zeros(size=(0,))

    if len(speaker_emotion_clip) > 0:
        speaker_emotion_clip = torch.stack(speaker_emotion_clip, dim=0)
    else:
        speaker_emotion_clip = torch.zeros(size=(0,))

    if len(speaker_3dmm_clip) > 0:
        speaker_3dmm_clip = torch.stack(speaker_3dmm_clip, dim=0)
    else:
        speaker_3dmm_clip = torch.zeros(size=(0,))

    if len(listener_video_clip) > 0:
        listener_video_clip = torch.stack(listener_video_clip, dim=0)
    else:
        listener_video_clip = torch.zeros(size=(0,))

    listener_emotion_clip = torch.stack(listener_emotion_clip, dim=0)
    listener_3dmm_clip = torch.stack(listener_3dmm_clip, dim=0)
    _, l, _3dmm_dim = listener_3dmm_clip.shape
    listener_3dmm_clip_personal = torch.stack(listener_3dmm_clip_personal, dim=0)
    l = listener_3dmm_clip_personal.shape[-2]
    listener_3dmm_clip_personal = listener_3dmm_clip_personal.reshape(-1, l, _3dmm_dim)

    if len(listener_reference) > 0:
        listener_reference = torch.stack(listener_reference, dim=0)
    else:
        listener_reference = torch.zeros(size=(0,))

    return (
        speaker_audio_clip,
        speaker_video_clip,
        speaker_emotion_clip,
        speaker_3dmm_clip,
        listener_video_clip,
        listener_emotion_clip,
        listener_3dmm_clip,
        listener_3dmm_clip_personal,
        listener_reference,
    )


class ReactionDatasetTrain(data.Dataset):
    def __init__(self, root_path, split, img_size=256, crop_size=224, num_person=16, num_sample=4, clip_length=751,
                 fps=25, load_audio=True, load_video_s=True, load_video_l=False, load_emotion_s=True,
                 load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=True, load_ref=True, k_appro=10):

        self._root_path = root_path
        self._num_person = num_person
        self._num_sample = num_sample
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self._data_path = os.path.join(self._root_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)

        neighbour_emotion_path = os.path.join(root_path, 'neighbour_emotion_' + split + '.npy')
        self.neighbour_emotion = np.load(neighbour_emotion_path)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

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
        self.neighbour_ids = {}
        for i, listener_id in enumerate(self.listener_ids):
            if listener_id not in self.neighbour_ids:
                self.neighbour_ids[listener_id] = [i]
            else:
                self.neighbour_ids[listener_id].append(i)

        self.data_list = [path for path in list(self._list_path.values[:, 1])] + [path for path in list(
            self._list_path.values[:, 2])]

        self.k_appro = k_appro

        self._len = len(self.data_list)  # 3186

    def __getitem__(self, index):
        speaker_line = self.neighbour_emotion[index]  # self.neighbour_emotion.shape: (3186, 3186)
        sim_speakers_index = np.where(speaker_line == True)[0]

        appro_listeners_index = sim_speakers_index
        if len(appro_listeners_index) > 1:
            new_list = []
            new_list.append(index)
            for e in sim_speakers_index:
                if e != index:
                    new_list.append(e)
            appro_listeners_index = new_list

        appro_listeners_ids = [self.listener_ids[idx] for idx in appro_listeners_index]
        appro_listeners = {}
        for i in range(len(appro_listeners_index)):
            if appro_listeners_ids[i] not in appro_listeners:
                appro_listeners[appro_listeners_ids[i]] = [appro_listeners_index[i]]
            else:
                appro_listeners[appro_listeners_ids[i]].append(appro_listeners_index[i])

        appro_listeners_index_personal = []
        for i, listener_id in enumerate(appro_listeners_ids):
            idx = appro_listeners_index[i]
            if len(appro_listeners[listener_id]) >= 2:
                indices = deepcopy(appro_listeners[listener_id])
                indices.remove(idx)
                selected_idx = random.choice(indices)
                appro_listeners_index_personal.append(selected_idx)
            else:
                whole_indices = deepcopy(self.neighbour_ids[listener_id])
                whole_indices.remove(idx)
                selected_idx = random.choice(whole_indices)
                appro_listeners_index_personal.append(selected_idx)

        if len(appro_listeners_index) < self.k_appro:
            random_indices = np.random.randint(0, len(appro_listeners_index), size=(self.k_appro,))
        else:
            random_indices = np.random.permutation(len(appro_listeners_index))[:self.k_appro]
        relative_indices_personal = [appro_listeners_index_personal[i] for i in random_indices]
        appro_listeners_index = [appro_listeners_index[j] for j in random_indices]

        total_length = 751
        cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_clip = torch.zeros(size=(0,))
        if self.load_video_s:
            speaker_video_path = os.path.join(self._video_path, self.speaker_path[index] + '.mp4')
            clip = []
            with open(speaker_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp, cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            speaker_video_clip = torch.stack(clip, dim=0)

        listener_video_clip = torch.zeros(size=(0,))
        listener_video_clip_personal = torch.zeros(size=(0,))
        if self.load_video_l:
            listener_video_path = os.path.join(self._video_path,
                                               self.listener_path[index] + '.mp4')
            clip = []
            with open(listener_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp, cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            # shape: [_clip_length, 3, 224, 224]
            listener_video_clip = torch.stack(clip, dim=0)

        speaker_audio_clip = torch.zeros(size=(0,))
        if self.load_audio:
            speaker_audio_path = os.path.join(self._audio_path, self.speaker_path[index] + '.wav')
            speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)

            # shape: [_clip_length, 78]
            speaker_audio_clip = torch.from_numpy(speaker_audio_clip[cp:cp + self._clip_length])

        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion_clip = torch.zeros(size=(0,))
        listener_emotion_clip_personal = torch.zeros(size=(0,))
        if self.load_emotion_l:
            selected_listener_emotion = []
            union_index = appro_listeners_index

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

            # shape: [k_appro, _clip_length, 25]
            listener_emotion_clip = torch.stack(selected_listener_emotion)

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

            listener_emotion_clip_personal = torch.stack(selected_listener_emotion)

        speaker_emotion_clip = torch.zeros(size=(0,))
        if self.load_emotion_s:
            speaker_emotion_path = os.path.join(self._emotion_path, self.speaker_path[index] + '.csv')

            if 'NoXI' in speaker_emotion_path:
                speaker_emotion_path = speaker_emotion_path.replace('Novice_video', 'P2')
                speaker_emotion_path = speaker_emotion_path.replace('Expert_video', 'P1')

            if 'Emotion/RECOLA/group' in speaker_emotion_path:
                speaker_emotion_path = speaker_emotion_path.replace('P25', 'P1')
                speaker_emotion_path = speaker_emotion_path.replace('P26', 'P2')
                speaker_emotion_path = speaker_emotion_path.replace('P41', 'P1')
                speaker_emotion_path = speaker_emotion_path.replace('P42', 'P2')
                speaker_emotion_path = speaker_emotion_path.replace('P45', 'P1')
                speaker_emotion_path = speaker_emotion_path.replace('P46', 'P2')

            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')

            # shape: [_clip_length, 25]
            speaker_emotion_clip = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                                   cp: cp + self._clip_length]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm_clip = torch.zeros(size=(0,))
        listener_3dmm_clip_personal = torch.zeros(size=(0,))
        if self.load_3dmm_l:
            selected_listener_3dmm = []
            union_index = appro_listeners_index

            for i in union_index:
                listener_3dmm_path = os.path.join(self._3dmm_path, self.listener_path[i] + '.npy')
                listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
                listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
                listener_3dmm = self._transform_3dmm(listener_3dmm)[0]
                selected_listener_3dmm.append(listener_3dmm)
            # shape: [k_appro, _clip_length, 25]
            listener_3dmm_clip = torch.stack(selected_listener_3dmm)

            selected_listener_3dmm = []
            union_index = relative_indices_personal

            for i in union_index:
                listener_3dmm_path = os.path.join(self._3dmm_path, self.listener_path[i] + '.npy')
                listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()

                listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
                listener_3dmm = self._transform_3dmm(listener_3dmm)[0]
                selected_listener_3dmm.append(listener_3dmm)

            # shape: [k_appro, _clip_length, 25]
            listener_3dmm_clip_personal = torch.stack(selected_listener_3dmm)

        speaker_3dmm_clip = torch.zeros(size=(0,))
        if self.load_3dmm_s:
            speaker_3dmm_path = os.path.join(self._3dmm_path, self.speaker_path[index] + '.npy')
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length]
            speaker_3dmm_clip = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = torch.zeros(size=(0,))
        if self.load_ref:
            listener_video_path = os.path.join(self._video_path, self.listener_path[index] + '.mp4')
            container = av.open(listener_video_path)  # read mp4 files

            for frame in container.decode(video=0):
                img = frame.to_image().convert('RGB')
                break

            # shape: [3, 224, 224]
            listener_reference = self._transform(img)

        return (
            speaker_audio_clip,
            speaker_video_clip,
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,
            listener_video_clip_personal,
            listener_emotion_clip,
            listener_emotion_clip_personal,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
        )

    def __len__(self):
        return self._len


class ReactionDatasetTest(data.Dataset):
    def __init__(self, root_path, split, img_size=256, crop_size=224, clip_length=751, fps=25, load_audio=True,
                 load_video_s=False, load_video_l=False, load_emotion_s=True, load_emotion_l=False, load_3dmm_s=True,
                 load_3dmm_l=True, load_ref=True):

        self._root_path = root_path
        self._clip_length = clip_length
        self._fps = fps
        self._split = split

        self._data_path = os.path.join(self._root_path, 'person_specific_' + self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, 'person_specific_' + self._split + '.csv'),
                                      header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)
        neighbour_emotion_path = os.path.join(
            root_path, 'person_specific_masked_neighbour_emotion_' + split + '.npy')

        self.neighbour_emotion = np.load(neighbour_emotion_path)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_3dmm_s = load_3dmm_s
        self.load_video_l = load_video_l
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

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

        self.data_list = [path for path in list(self._list_path.values[:, 1])] + [path for path in list(
            self._list_path.values[:, 2])]

        self._len = len(self.data_list)  # 3092

    def __getitem__(self, index):
        speaker_line = self.neighbour_emotion[index]  # self.neighbour_emotion.shape: (3186, 3186)
        sim_speakers_index = np.where(speaker_line == True)[0]
        listener_personal_path = os.path.join('/'.join(self.listener_path[index].split('/')[:3]), '1')

        appro_listeners_index = sim_speakers_index
        if len(appro_listeners_index) > 1:
            new_list = []
            new_list.append(index)
            for e in sim_speakers_index:
                if e != index:
                    new_list.append(e)
            appro_listeners_index = new_list

        total_length = 751
        cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_clip = torch.zeros(size=(0,))
        if self.load_video_s:
            speaker_video_path = os.path.join(self._video_path, self.speaker_path[index] + '.mp4')
            clip = []
            with open(speaker_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp, cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            speaker_video_clip = torch.stack(clip, dim=0)

        listener_video_clip = torch.zeros(size=(0,))
        if self.load_video_l:
            listener_video_path = os.path.join(self._video_path,
                                               self.listener_path[index] + '.mp4')
            clip = []
            with open(listener_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp, cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            listener_video_clip = torch.stack(clip, dim=0)

        speaker_audio_clip = torch.zeros(size=(0,))
        if self.load_audio:
            speaker_audio_path = os.path.join(self._audio_path, self.speaker_path[index] + '.wav')
            speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)
            # shape: [_clip_length, 78]
            speaker_audio_clip = torch.from_numpy(speaker_audio_clip[cp:cp + self._clip_length])

        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion_clip = torch.zeros(size=(0,))
        listener_emotion_clip_personal = torch.zeros(size=(0,))
        if self.load_emotion_l:
            selected_listener_emotion = []
            union_index = appro_listeners_index[0:1]

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

            # shape: [_clip_length, 25]
            listener_emotion_clip = selected_listener_emotion[0]

            listener_emotion_path = os.path.join(self._emotion_path, listener_personal_path + '.csv')
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
            listener_emotion_clip_personal = emotion.unsqueeze(0).expand(10, -1, -1)

        speaker_emotion_clip = torch.zeros(size=(0,))
        if self.load_emotion_s:
            speaker_emotion_path = os.path.join(self._emotion_path, self.speaker_path[index] + '.csv')

            if 'NoXI' in speaker_emotion_path:
                speaker_emotion_path = speaker_emotion_path.replace('Novice_video', 'P2')
                speaker_emotion_path = speaker_emotion_path.replace('Expert_video', 'P1')

            if 'Emotion/RECOLA/group' in speaker_emotion_path:
                speaker_emotion_path = speaker_emotion_path.replace('P25', 'P1')
                speaker_emotion_path = speaker_emotion_path.replace('P26', 'P2')
                speaker_emotion_path = speaker_emotion_path.replace('P41', 'P1')
                speaker_emotion_path = speaker_emotion_path.replace('P42', 'P2')
                speaker_emotion_path = speaker_emotion_path.replace('P45', 'P1')
                speaker_emotion_path = speaker_emotion_path.replace('P46', 'P2')

            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')

            # shape: [_clip_length, 25]
            speaker_emotion_clip = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                                   cp: cp + self._clip_length]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm_clip = torch.zeros(size=(0,))
        listener_3dmm_clip_personal = torch.zeros(size=(0,))
        if self.load_3dmm_l:
            selected_listener_3dmm = []
            union_index = appro_listeners_index[0:1]

            for i in union_index:
                listener_3dmm_path = os.path.join(self._3dmm_path, self.listener_path[i] + '.npy')
                listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
                listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
                listener_3dmm = self._transform_3dmm(listener_3dmm)[0]
                selected_listener_3dmm.append(listener_3dmm)

            # shape: [_clip_length, 25]
            listener_3dmm_clip = selected_listener_3dmm[0]
            listener_3dmm_path = os.path.join(self._3dmm_path, listener_personal_path + '.npy')
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()

            listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
            listener_3dmm = self._transform_3dmm(listener_3dmm)[0]
            listener_3dmm_clip_personal = listener_3dmm.unsqueeze(0).expand(10, -1, -1)

        speaker_3dmm_clip = torch.zeros(size=(0,))
        if self.load_3dmm_s:
            speaker_3dmm_path = os.path.join(self._3dmm_path, self.speaker_path[index] + '.npy')
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length]
            speaker_3dmm_clip = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = torch.zeros(size=(0,))
        if self.load_ref:
            listener_video_path = os.path.join(self._video_path, self.listener_path[index] + '.mp4')
            container = av.open(listener_video_path)  # read mp4 files

            for frame in container.decode(video=0):
                img = frame.to_image().convert('RGB')
                # img = self._transform(img)
                break

            # shape: [3, 224, 224]
            listener_reference = self._transform(img)

        return (
            speaker_audio_clip,
            speaker_video_clip,
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,
            listener_emotion_clip,
            listener_emotion_clip_personal,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
        )

    def __len__(self):
        return self._len


def get_dataloader(conf):
    assert conf.split in ["train", "val", "test"], "split must be in [train, val, test]"
    print('==> Preparing data for {}...'.format(conf.split) + '\n')

    if conf.split in ["train", "val"]:  # train or validation
        custom_collate = custom_collate_train
        dataset = ReactionDatasetTrain(
            root_path=conf.dataset_path,
            split=conf.split,
            num_person=conf.num_person,
            num_sample=conf.num_sample,
            img_size=conf.img_size,
            crop_size=conf.crop_size,
            clip_length=conf.clip_length,
            fps=conf.fps,
            load_audio=conf.load_audio,
            load_video_s=conf.load_video_s,
            load_video_l=conf.load_video_l,
            load_emotion_s=conf.load_emotion_s,
            load_emotion_l=conf.load_emotion_l,
            load_3dmm_s=conf.load_3dmm_s,
            load_3dmm_l=conf.load_3dmm_l,
            load_ref=conf.load_ref,
            k_appro=conf.k_appro
        )
    else:
        custom_collate = custom_collate_test
        dataset = ReactionDatasetTest(
            root_path=conf.dataset_path,
            split=conf.split,
            img_size=conf.img_size,
            crop_size=conf.crop_size,
            clip_length=conf.clip_length,
            fps=conf.fps,
            load_audio=conf.load_audio,
            load_video_s=conf.load_video_s,
            load_video_l=conf.load_video_l,
            load_emotion_s=conf.load_emotion_s,
            load_emotion_l=conf.load_emotion_l,
            load_3dmm_s=conf.load_3dmm_s,
            load_3dmm_l=conf.load_3dmm_l,
            load_ref=conf.load_ref,
        )

    dataloader = DataLoader(dataset=dataset,
                            collate_fn=custom_collate,
                            batch_size=conf.batch_size,
                            shuffle=conf.shuffle,
                            num_workers=conf.num_workers)
    return dataloader
