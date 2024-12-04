import os
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import random
import pandas as pd
from PIL import Image
import soundfile as sf
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


def extract_fbank_features(audio_path, num_mel_bins=128, target_length=750, skip_norm=False,
                           norm_mean=-4.2677393, norm_std=4.5689974, noise=False):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0,
                                              frame_shift=10)
    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if not skip_norm:
        fbank = (fbank - norm_mean) / (norm_std * 2)

    if noise == True:
        fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

    return fbank


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


class ReactionDataset(data.Dataset):
    def __init__(self, root_path, split, img_size=256, crop_size=224, clip_length=751, fps=25, load_audio=True,
                 load_video=True, load_emotion=True, load_3dmm=True):

        self._root_path = root_path
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self._data_path = os.path.join(self._root_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)

        self.load_audio = load_audio
        self.load_video = load_video
        self.load_emotion = load_emotion
        self.load_3dmm = load_3dmm

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
            self._list_path.values[:, 2])]  # the data_list is actually the same as speaker_path

        self._len = len(self.data_list)  # 3186

    def __getitem__(self, index):
        total_length = 751
        cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0

        # ========================= Load video clip ==========================
        video_clip = torch.zeros(size=(0,))
        if self.load_video:
            video_path = os.path.join(self._video_path, self.speaker_path[index] + '.mp4')
            clip = []
            with open(video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp, cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)

            # shape: [_clip_length, 3, 224, 224]
            video_clip = torch.stack(clip, dim=0)

        # ========================= Load audio clip ==========================
        audio_clip = torch.zeros(size=(0,))
        if self.load_audio:
            audio_path = os.path.join(self._audio_path, self.speaker_path[index] + '.wav')
            audio_clip = extract_audio_features(audio_path, self._fps, total_length)

            # shape: [_clip_length, 78]
            audio_clip = torch.from_numpy(audio_clip[cp:cp + self._clip_length])

        # ========================= load emotion clip ==========================
        emotion_clip = torch.zeros(size=(0,))
        if self.load_emotion:
            emotion_path = os.path.join(self._emotion_path, self.speaker_path[index] + '.csv')

            if 'NoXI' in emotion_path:
                emotion_path = emotion_path.replace('Novice_video', 'P2')
                emotion_path = emotion_path.replace('Expert_video', 'P1')

            if 'Emotion/RECOLA/group' in emotion_path:
                emotion_path = emotion_path.replace('P25', 'P1')
                emotion_path = emotion_path.replace('P26', 'P2')
                emotion_path = emotion_path.replace('P41', 'P1')
                emotion_path = emotion_path.replace('P42', 'P2')
                emotion_path = emotion_path.replace('P45', 'P1')
                emotion_path = emotion_path.replace('P46', 'P2')

            emotion = pd.read_csv(emotion_path, header=None, delimiter=',')

            # shape: [_clip_length, 25]
            emotion_clip = torch.from_numpy(np.array(emotion.drop(0)).astype(np.float32))[
                           cp: cp + self._clip_length]

        # ========================= load 3dmm clip =========================
        _3dmm_clip = torch.zeros(size=(0,))
        if self.load_3dmm:
            _3dmm_path = os.path.join(self._3dmm_path, self.speaker_path[index] + '.npy')
            _3dmm = torch.FloatTensor(np.load(_3dmm_path)).squeeze()
            _3dmm = _3dmm[cp: cp + self._clip_length]
            _3dmm_clip = self._transform_3dmm(_3dmm)[0]

        return audio_clip, video_clip, emotion_clip, _3dmm_clip

    def __len__(self):
        return self._len


def get_dataloader(conf, dataset_path, split):
    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    print('==> Preparing data for {}...'.format(split) + '\n')

    dataset = ReactionDataset(root_path=dataset_path,
                              split=split,
                              img_size=conf.dataset.img_size,
                              crop_size=conf.dataset.crop_size,
                              clip_length=conf.dataset.clip_length,
                              fps=conf.dataset.fps,
                              load_audio=conf.dataset.load_audio,
                              load_video=conf.dataset.load_video,
                              load_emotion=conf.dataset.load_emotion,
                              load_3dmm=conf.dataset.load_3dmm)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=conf.dataset.batch_size,
                            shuffle=conf.dataset.shuffle,
                            num_workers=conf.dataset.num_workers)

    return dataloader