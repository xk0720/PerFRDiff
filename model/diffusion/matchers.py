import torch
import torch.nn as nn
import os
import model as module_arch
from model.diffusion.mlp_diffae import MLPSkipNet, Activation
from model.diffusion.diffusion_prior.transformer_prior import DiffusionPriorNetwork
from model.diffusion.diffusion_decoder.transformer_denoiser import TransformerDenoiser
from model.diffusion.gaussian_diffusion import PriorLatentDiffusion, DecoderLatentDiffusion
from model.diffusion.resample import UniformSampler
from utils.util import load_config_from_file, checkpoint_load
from einops import rearrange


class BaseLatentModel(nn.Module):
    def __init__(self, device, cfg, emb_preprocessing=False, freeze_encoder=True):
        super(BaseLatentModel, self).__init__()
        self.emb_preprocessing = emb_preprocessing
        self.freeze_encoder = freeze_encoder
        def_dtype = torch.get_default_dtype()

        self.latent_embedder = getattr(module_arch, cfg.latent_embedder.type)(cfg.latent_embedder.args)
        model_path = cfg.latent_embedder.checkpoint_path
        assert os.path.exists(model_path), (
            "Miss checkpoint for model: {}.".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.latent_embedder.load_state_dict(state_dict)

        self.audio_encoder = getattr(module_arch, cfg.audio_encoder.type)(**cfg.audio_encoder.args)
        model_path = cfg.audio_encoder.get("checkpoint_path", None)
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.audio_encoder.load_state_dict(state_dict)

        self.person_encoder = getattr(module_arch, cfg.person_specific.type)(device, **cfg.person_specific.args)
        model_path = cfg.person_specific.checkpoint_path
        assert os.path.exists(model_path), (
            "Miss checkpoint for model: {}.".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.person_encoder.load_state_dict(state_dict)

        self.latent_3dmm_embedder = getattr(module_arch, cfg.latent_3dmm_embedder.type)(cfg.latent_3dmm_embedder.args)
        model_path = cfg.latent_3dmm_embedder.checkpoint_path
        assert os.path.exists(model_path), (
            "Miss checkpoint for latent embedder: {}.".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.latent_3dmm_embedder.load_state_dict(state_dict)

        if self.freeze_encoder:
            for para in self.latent_embedder.parameters():
                para.requires_grad = False
            for para in self.latent_3dmm_embedder.parameters():
                para.requires_grad = False
            for para in self.person_encoder.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype)

        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class PriorLatentMatcher(BaseLatentModel):
    def __init__(self, conf, device):
        cfg = conf.diffusion_prior.args
        super(PriorLatentMatcher, self).__init__(
            device,
            conf,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
        )

        self.window_size = cfg.window_size
        self.token_len = cfg.token_len
        self.init_params = {
            "audio_dim": cfg.get("audio_dim", 78),
            "window_size": cfg.get("window_size", 50),
            "_3dmm_dim": cfg.get("_3dmm_dim", 58),
            "speaker_emb_dim": cfg.get("speaker_emb_dim", 512),
            "latent_dim": cfg.get("latent_dim", 512),
            "depth": cfg.get("depth", 6),
            "num_time_layers": cfg.get("num_time_layers", 2),
            "num_time_embeds": cfg.get("num_time_embeds", 1),
            "num_time_emb_channels": cfg.get("num_time_emb_channels", 64),
            "time_last_act": cfg.get("time_last_act", False),
            "use_learned_query": cfg.get("use_learned_query=True", True),
            "s_audio_cond_drop_prob": cfg.get("s_audio_cond_drop_prob", 0.5),
            "s_latentemb_cond_drop_prob": cfg.get("s_latentemb_cond_drop_prob", 0.5),
            "s_3dmm_cond_drop_prob": cfg.get("s_3dmm_cond_drop_prob", 0.5),
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "dim_head": cfg.get("dim_head", 64),
            "ff_mult": cfg.get("ff_mult", 4),
            "norm_in": cfg.get("norm_in", False),
            "norm_out": cfg.get("norm_out", True),
            "attn_dropout": cfg.get("attn_dropout", 0.0),
            "ff_dropout": cfg.get("ff_dropout", 0.0),
            "final_proj": cfg.get("final_proj", True),
            "normformer": cfg.get("normformer", False),
            "rotary_emb": cfg.get("rotary_emb", True),
        }
        self.model = DiffusionPriorNetwork(**self.init_params)

        self.mode = conf.mode
        if self.mode == "test":  # load checkpoints
            checkpoint_load(conf, self.model, device, checkpoint_path=None)
            # print(f'Successfully load checkpoint: {self.model.get_model_name()}')

        self.prior_diffusion = PriorLatentDiffusion(
            conf.diffusion_prior.args,
            conf.diffusion_prior.scheduler,
            conf.diffusion_prior.scheduler.num_train_timesteps,
            conf.diffusion_prior.scheduler.num_inference_timesteps,
        )

        self.schedule_sampler = UniformSampler(self.prior_diffusion)

        self.k = conf.diffusion_prior.scheduler.k

    def _forward(
            self,
            speaker_audio=None,
            speaker_emotion=None, 
            speaker_3dmm=None,
            listener_emotion=None,
            listener_3dmm=None,
            **kwargs,
    ):

        batch_size, seq_len, d = speaker_audio.shape
        emo_dim = listener_emotion.shape[-1]
        _3dmm_dim = listener_3dmm.shape[-1]

        if self.mode in ["train", "val"]: 
            window_start = torch.randint(0, seq_len - self.window_size, (1,), device=speaker_audio.device)
            window_end = window_start + self.window_size

            s_audio_selected = speaker_audio[:, window_start:window_end]
            s_emotion_selected = speaker_emotion[:, window_start:window_end]
            s_3dmm_selected = speaker_3dmm[:, window_start:window_end]
            l_emotion_selected = listener_emotion[:, window_start:window_end]

            with torch.no_grad():
                s_audio_encodings = self.audio_encoder._encode(s_audio_selected)
                s_audio_encodings = s_audio_encodings.repeat_interleave(self.k, dim=0)

                x_start = self.latent_embedder.encode(l_emotion_selected).unsqueeze(1)  # (..., l_3dmm_selected)

                s_latent_embed = self.latent_embedder.encode(s_emotion_selected).unsqueeze(1)  # (..., s_3dmm_selected)
                s_latent_embed = s_latent_embed.repeat_interleave(self.k, dim=0)

                s_3dmm_encodings = s_3dmm_selected.repeat_interleave(self.k, dim=0)

                model_kwargs = {"speaker_audio_encodings": s_audio_encodings,
                                "speaker_latent_emb": s_latent_embed,
                                "speaker_3dmm_encodings": s_3dmm_encodings}

            t, weights = self.schedule_sampler.sample(batch_size, speaker_audio.device)
            output_pred = self.prior_diffusion.denoise(self.model, x_start, t, model_kwargs=model_kwargs)

            return output_pred

        else:
            assert seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
            diff_batch = batch_size * (seq_len // self.window_size)

            s_audio = speaker_audio.reshape(diff_batch, self.window_size, d)
            s_emotion = speaker_emotion.reshape(diff_batch, self.window_size, emo_dim)
            s_3dmm = speaker_3dmm.reshape(diff_batch, self.window_size, _3dmm_dim)

            listener_emotion = listener_emotion.reshape(-1, self.k, (seq_len // self.window_size),
                                                        self.window_size, emo_dim)
            listener_emotion = listener_emotion.transpose(1, 2).contiguous()
            l_emotion = listener_emotion.reshape(-1, self.window_size, emo_dim)

            with torch.no_grad():
                s_audio_encodings = self.audio_encoder._encode(s_audio)
                s_audio_encodings = s_audio_encodings.repeat_interleave(self.k, dim=0)

                s_latent_embed = self.latent_embedder.encode(s_emotion).unsqueeze(1)  # (..., s_3dmm)
                s_latent_embed = s_latent_embed.repeat_interleave(self.k, dim=0)

                speaker_3dmm_encodings = s_3dmm.repeat_interleave(self.k, dim=0)

                speaker_emotion_encodings = s_emotion.repeat_interleave(self.k, dim=0)

                listener_gt = self.latent_embedder.encode(l_emotion).unsqueeze(1)

                model_kwargs = {"speaker_audio_encodings": s_audio_encodings,
                                "speaker_latent_emb": s_latent_embed,
                                "speaker_3dmm_encodings": speaker_3dmm_encodings}

            output = [output for output in self.prior_diffusion.ddim_sample_loop_progressive(
                matcher=self,
                model=self.model,
                model_kwargs=model_kwargs,
                gt=listener_gt,
            )][-1] 

            output_prior = output["sample_enc"]
            model_kwargs = model_kwargs.copy()
            model_kwargs["speaker_emotion_encodings"] = speaker_emotion_encodings

            return output_prior, model_kwargs

    def forward_prior(self,
                      speaker_audio=None,
                      speaker_emotion_input=None,
                      speaker_3dmm_input=None, 
                      listener_emotion_input=None, 
                      listener_3dmm_input=None, 
                      **kwargs):

        if self.mode in ["train", "val"]:
            # conditions
            speaker_audio_shifted = speaker_audio[:, :-self.window_size]
            speaker_emotion_shifted = speaker_emotion_input[:, :-self.window_size]
            speaker_3dmm_shifted = speaker_3dmm_input[:, :-self.window_size]
            
            listener_emotion_shifted = listener_emotion_input[:, self.window_size:]
            listener_3dmm_shifted = listener_3dmm_input[:, self.window_size:]
        else:
            speaker_audio_shifted = torch.cat([torch.zeros_like(speaker_audio[:, :self.window_size]),
                                               speaker_audio[:, :-self.window_size]], dim=1)
            speaker_emotion_shifted = torch.cat([torch.zeros_like(speaker_emotion_input[:, :self.window_size]),
                                                 speaker_emotion_input[:, :-self.window_size]], dim=1)
            speaker_3dmm_shifted = torch.cat([torch.zeros_like(speaker_3dmm_input[:, :self.window_size]),
                                              speaker_3dmm_input[:, :-self.window_size]], dim=1)
            listener_emotion_shifted = listener_emotion_input
            listener_3dmm_shifted = listener_3dmm_input

        return self._forward(speaker_audio_shifted,
                             speaker_emotion_shifted,
                             speaker_3dmm_shifted,
                             listener_emotion_shifted,
                             listener_3dmm_shifted,
                             **kwargs)

    def forward(self, **kwargs):
        return self.forward_prior(**kwargs)


class DecoderLatentMatcher(BaseLatentModel):
    def __init__(self, conf, device):
        cfg = conf.diffusion_decoder.args
        super(DecoderLatentMatcher, self).__init__(
            device,
            conf,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
        )

        self.window_size = cfg.window_size
        self.token_len = cfg.token_len
        self.emotion_dim = cfg.get("nfeats", 25)
        self.encode_emotion = cfg.get("encode_emotion", False)
        self.encode_3dmm = cfg.get("encode_3dmm", False)

        self.init_params = {
            "encode_emotion": self.encode_emotion,
            "encode_3dmm": self.encode_3dmm,
            "ablation_skip_connection": cfg.get("ablation_skip_connection", True),
            "nfeats": cfg.get("nfeats", 25),
            "latent_dim": cfg.get("latent_dim", 512),
            "ff_size": cfg.get("ff_size", 1024),
            "num_layers": cfg.get("num_layers", 6),
            "num_heads": cfg.get("num_heads", 4),
            "dropout": cfg.get("dropout", 0.1),
            "normalize_before": cfg.get("normalize_before", False),
            "activation": cfg.get("activation", "gelu"),
            "flip_sin_to_cos": cfg.get("flip_sin_to_cos", True),
            "return_intermediate_dec": cfg.get("return_intermediate_dec", False),
            "position_embedding": cfg.get("position_embedding", "learned"),
            "arch": cfg.get("arch", "trans_enc"),
            "freq_shift": cfg.get("freq_shift", 0),
            "time_encoded_dim": cfg.get("time_encoded_dim", 64),
            "s_audio_dim": cfg.get("s_audio_dim", 78),
            "s_audio_scale": cfg.get("s_audio_scale", cfg.get("latent_dim", 512) ** -0.5),
            "s_emotion_dim": cfg.get("s_emotion_dim", 25),
            "l_embed_dim": cfg.get("l_embed_dim", 512),
            "s_embed_dim": cfg.get("s_embed_dim", 512),
            "personal_emb_dim": cfg.get("personal_emb_dim", 512),
            "s_3dmm_dim": cfg.get("s_3dmm_dim", 58),
            "concat": cfg.get("concat", "concat_first"),
            "condition_concat": cfg.get("condition_concat", "token_concat"),
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "l_latent_embed_drop_prob": cfg.get("l_latent_embed_drop_prob", 0.2),
            "l_personal_embed_drop_prob": cfg.get("l_personal_embed_drop_prob", 0.2),
            "s_audio_enc_drop_prob": cfg.get("s_audio_enc_drop_prob", 0.2),
            "s_latent_embed_drop_prob": cfg.get("s_latent_embed_drop_prob", 0.2),
            "s_3dmm_enc_drop_prob": cfg.get("s_3dmm_enc_drop_prob", 0.2),
            "s_emotion_enc_drop_prob": cfg.get("s_emotion_enc_drop_prob", 0.2),
            "past_l_emotion_drop_prob": cfg.get("past_l_emotion_drop_prob", 0.2),
        }
        self.use_past_frames = cfg.get("use_past_frames", False)

        self.model = TransformerDenoiser(**self.init_params)
        self.mode = conf.mode
        if self.mode == "test":
            checkpoint_load(conf, self.model, device, checkpoint_path=None)
            # print(f'Successfully load checkpoint: {self.model.get_model_name()}')

        self.decoder_diffusion = DecoderLatentDiffusion(
            conf.diffusion_decoder.scheduler,
            conf.diffusion_decoder.scheduler.num_train_timesteps,
            conf.diffusion_decoder.scheduler.num_inference_timesteps,
        )
        self.schedule_sampler = UniformSampler(self.decoder_diffusion)

        self.k = conf.diffusion_decoder.scheduler.k

    def _forward(
            self,
            speaker_audio=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            listener_3dmm_input=None,
            listener_personal_input=None,
            speaker_audio_encodings=None,
            speaker_latent_emb=None,
            listener_latent_embed=None,
    ):

        if self.mode in ["train", "val"]:
            speaker_audio_shifted = speaker_audio[:, :-self.window_size]
            batch_size, seq_len, d = speaker_audio_shifted.shape
            speaker_emotion_shifted = speaker_emotion_input[:, :-self.window_size]
            speaker_3dmm_shifted = speaker_3dmm_input[:, :-self.window_size]

            listener_emotion_shifted = listener_emotion_input[:, self.window_size:]
            past_listener_emotion = listener_emotion_input[:, :-self.window_size]

            window_start = torch.randint(0, seq_len - self.window_size, (1,), device=speaker_audio.device)
            window_end = window_start + self.window_size

            s_audio_selected = speaker_audio_shifted[:, window_start:window_end]
            s_emotion_selected = speaker_emotion_shifted[:, window_start:window_end]
            s_3dmm_selected = speaker_3dmm_shifted[:, window_start:window_end]

            l_emotion_selected = listener_emotion_shifted[:, window_start:window_end]
            l_personal_input = listener_personal_input
            past_listener_emotion = past_listener_emotion[:, window_start:window_end]

            x_start_selected = l_emotion_selected

            with torch.no_grad():
                personal_embed = self.person_encoder.forward(l_personal_input)[0].unsqueeze(1)
                
                l_latent_embed = self.latent_embedder.encode(l_emotion_selected).unsqueeze(1)

                s_audio_encodings = self.audio_encoder._encode(s_audio_selected)
                s_audio_encodings = s_audio_encodings.repeat_interleave(self.k, dim=0)

                s_latent_embed = self.latent_embedder.encode(s_emotion_selected).unsqueeze(1)
                s_latent_embed = s_latent_embed.repeat_interleave(self.k, dim=0)

                if self.encode_3dmm:
                    s_3dmm_selected = self.latent_3dmm_embedder.get_encodings(s_3dmm_selected)
                s_3dmm_encodings = s_3dmm_selected.repeat_interleave(self.k, dim=0)

                if self.encode_emotion:
                    s_emotion_selected = self.latent_embedder.get_encodings(s_emotion_selected)
                s_emotion_encodings = s_emotion_selected.repeat_interleave(self.k, dim=0)

                model_kwargs = {"listener_latent_embed": l_latent_embed,
                                "listener_personal_embed": personal_embed,
                                "speaker_audio_encodings": s_audio_encodings,
                                "speaker_latent_embed": s_latent_embed,
                                "speaker_3dmm_encodings": s_3dmm_encodings,
                                "speaker_emotion_encodings": s_emotion_encodings,
                                "past_listener_emotion": past_listener_emotion}

            t, _ = self.schedule_sampler.sample(batch_size, x_start_selected.device)
            timesteps = t.long()

            output_whole = self.decoder_diffusion.denoise(self.model, x_start_selected, timesteps,
                                                          model_kwargs=model_kwargs)

            return output_whole

        else:
            diff_batch = speaker_latent_emb.shape[0]
            seq_len = (self.token_len // self.window_size)
            batch_size = diff_batch // (seq_len * self.k)

            with torch.no_grad():
                if self.encode_3dmm:
                    speaker_3dmm_input = self.latent_3dmm_embedder.get_encodings(speaker_3dmm_input)
                if self.encode_emotion:
                    speaker_emotion_input = self.latent_embedder.get_encodings(speaker_emotion_input)

                personal_embed, _ = self.person_encoder.forward(listener_personal_input)

            dim = personal_embed.shape[-1]
            personal_embed = personal_embed.reshape(-1, self.k, dim)
            personal_embed = personal_embed.repeat(1, seq_len, 1)
            personal_embed = personal_embed.reshape(-1, dim)

            if self.use_past_frames:
                listener_latent_embed = listener_latent_embed.reshape(batch_size, seq_len, self.k, 1, -1)
                personal_embed = personal_embed.reshape(batch_size, seq_len, self.k, -1)
                speaker_latent_emb = speaker_latent_emb.reshape(batch_size, seq_len, self.k, 1, -1)
                speaker_audio_encodings = speaker_audio_encodings.reshape(
                    batch_size, seq_len, self.k, self.window_size, -1)
                speaker_3dmm_encodings = speaker_3dmm_input.reshape(
                    batch_size, seq_len, self.k, self.window_size, -1)
                speaker_emotion_encodings = speaker_emotion_input.reshape(
                    batch_size, seq_len, self.k, self.window_size, -1)

                past_listener_emotion = torch.zeros(
                    size=(batch_size * self.k, self.window_size, self.emotion_dim)
                ).to(device=listener_latent_embed.device)
                output_listener_emotion = torch.zeros(
                    size=(seq_len, batch_size, self.k, self.window_size, self.emotion_dim)
                ).to(device=listener_latent_embed.device)

                for i in range(seq_len):
                    model_kwargs = {
                        "listener_latent_embed": listener_latent_embed[:, i].reshape(batch_size * self.k, 1, -1),
                        "listener_personal_embed": personal_embed[:, i].reshape(batch_size * self.k, 1, -1),
                        "speaker_audio_encodings": speaker_audio_encodings[:, i].reshape(
                            batch_size * self.k, self.window_size, -1),
                        "speaker_latent_embed": speaker_latent_emb[:, i].reshape(batch_size * self.k, 1, -1),
                        "speaker_3dmm_encodings": speaker_3dmm_encodings[:, i].reshape(
                            batch_size * self.k, self.window_size, -1),
                        "speaker_emotion_encodings": speaker_emotion_encodings[:, i].reshape(
                            batch_size * self.k, self.window_size, -1),
                        "past_listener_emotion": past_listener_emotion,
                    }

                    with torch.no_grad():
                        output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                            matcher=self,
                            model=self.model,
                            model_kwargs=model_kwargs,
                            shape=past_listener_emotion.shape,
                        )][-1]

                    past_listener_emotion = output["sample_enc"]
                    output_listener_emotion[i, :, :, :, :] = output["sample_enc"].reshape(
                        batch_size, self.k, self.window_size, self.emotion_dim
                    )
                output_listener_emotion = output_listener_emotion.permute(1, 0, 2, 3, 4).contiguous()

            else:
                model_kwargs = {
                    "listener_latent_embed": listener_latent_embed,
                    "listener_personal_embed": personal_embed.unsqueeze(-2),
                    "speaker_audio_encodings": speaker_audio_encodings,
                    "speaker_latent_embed": speaker_latent_emb,
                    "speaker_3dmm_encodings": speaker_3dmm_input,
                    "speaker_emotion_encodings": speaker_emotion_input,
                }

                # online inference
                with torch.no_grad():
                    output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                        matcher=self,
                        model=self.model,
                        model_kwargs=model_kwargs,
                        shape=(diff_batch, self.window_size, self.emotion_dim),
                    )][-1]

                output_listener_emotion = output["sample_enc"]

            output_listener_emotion = output_listener_emotion.reshape(
                batch_size, seq_len, self.k, self.window_size, self.emotion_dim)
            output_listener_emotion = output_listener_emotion.transpose(1, 2).contiguous()
            output_listener_emotion = output_listener_emotion.reshape(batch_size, self.k, -1, self.emotion_dim)
            output_whole = {"prediction_emotion": output_listener_emotion}

            return output_whole

    def forward(self, **kwargs):
        return self._forward(**kwargs)


class LatentMatcher(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.mode = cfg.mode
        self.diffusion_prior = PriorLatentMatcher(cfg, device=device)
        self.diffusion_decoder = DecoderLatentMatcher(cfg, device=device)

    def forward(
            self,
            speaker_audio=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            listener_3dmm_input=None,
            listener_personal_input=None,
    ):

        if self.mode in ["train", "val"]:
            output_prior = self.diffusion_prior.forward(
                speaker_audio=speaker_audio,
                speaker_emotion_input=speaker_emotion_input,
                speaker_3dmm_input=speaker_3dmm_input,
                listener_emotion_input=listener_emotion_input,
                listener_3dmm_input=listener_3dmm_input
            )
            output_decoder = self.diffusion_decoder.forward(
                speaker_audio=speaker_audio,
                speaker_emotion_input=speaker_emotion_input,
                speaker_3dmm_input=speaker_3dmm_input,
                listener_emotion_input=listener_emotion_input,
                listener_personal_input=listener_personal_input,
            )

        else:
            output_prior, model_kwargs = self.diffusion_prior.forward(
                speaker_audio=speaker_audio,
                speaker_emotion_input=speaker_emotion_input,
                speaker_3dmm_input=speaker_3dmm_input,
                listener_emotion_input=listener_emotion_input, 
                listener_3dmm_input=listener_3dmm_input
            )
            
            speaker_latent_embed = model_kwargs["speaker_latent_emb"]
            speaker_audio_encodings = model_kwargs["speaker_audio_encodings"]
            speaker_3dmm_encodings = model_kwargs["speaker_3dmm_encodings"]
            speaker_emotion_encodings = model_kwargs[
                "speaker_emotion_encodings"]

            output_decoder = self.diffusion_decoder.forward(
                speaker_emotion_input=speaker_emotion_encodings,
                speaker_3dmm_input=speaker_3dmm_encodings,
                listener_personal_input=listener_personal_input,
                speaker_audio_encodings=speaker_audio_encodings,
                speaker_latent_emb=speaker_latent_embed,
                listener_latent_embed=output_prior,
            )

        return output_prior, output_decoder

    def obtain_shapes(self, modified_layers):
        shape_dict = {}
        for name, module in self.named_modules():
            if name in modified_layers:
                if 'linear' in name or 'to_emotion' in name:
                    shape_dict[name] = torch.tensor(module.weight.size())
                elif 'multihead_attn' in name or 'self_attn' in name:
                    shape_dict[name] = torch.tensor(module.in_proj_weight.size())
            else:
                continue
        return shape_dict
