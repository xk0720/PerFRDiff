import os
import logging
import torch
import argparse
from torch import optim
from functools import partial
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataset.dataset import get_dataloader
import model.diffusion.utils.losses as module_loss
from torch.utils.tensorboard import SummaryWriter
from utils.util import load_config, init_seed, get_logging_path, get_tensorboard_path, AverageMeter, save_checkpoint, \
    get_lr, collect_grad_stats
import model as module_arch


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--writer", type=bool, help="whether use tensorboard", default=True)
    parser.add_argument("--config", type=str, help="config path", default='./configs/diffusion_model.yaml')
    args = parser.parse_args()
    return args


def train(cfg, model, data_loader, optimizer, scheduler, criterion, epoch, writer, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()
    temporal_losses = AverageMeter()

    model.train()
    for batch_idx, (
            speaker_audio_clip,
            speaker_video_clip,
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
    ) in enumerate(tqdm(data_loader)):

        batch_size = speaker_audio_clip.shape[0]
        (speaker_audio_clip,
         speaker_emotion_clip,
         speaker_3dmm_clip,
         listener_emotion_clip,
         listener_3dmm_clip,
         listener_3dmm_clip_personal,
         listener_reference) = \
            (speaker_audio_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device),
             listener_reference.to(device))

        output_prior, output_decoder = model(
            speaker_audio=speaker_audio_clip,
            speaker_emotion_input=speaker_emotion_clip,
            speaker_3dmm_input=speaker_3dmm_clip,
            listener_emotion_input=listener_emotion_clip,
            listener_3dmm_input=listener_3dmm_clip,
            listener_personal_input=listener_3dmm_clip_personal,
        )

        output = criterion(output_prior, output_decoder)
        loss = output["loss"]
        temporal_loss = output["temporal_loss"]
        diff_prior_loss = output["encoded"]
        diff_decoder_loss = output["decoded"]

        iteration = batch_idx + len(data_loader) * epoch
        if writer is not None:
            writer.add_scalar("Train/whole_loss", loss.data.item(), iteration)
            writer.add_scalar("Train/diff_prior_loss", diff_prior_loss.data.item(), iteration)
            writer.add_scalar("Train/diff_decoder_loss", diff_decoder_loss.data.item(), iteration)
            writer.add_scalar("Train/temporal_loss", temporal_loss.data.item(), iteration)

        whole_losses.update(loss.data.item(), batch_size)
        diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
        diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)
        temporal_losses.update(temporal_loss.data.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        if cfg.trainer.clip_grad:
            clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

    if scheduler is not None and (epoch + 1) >= 5:
        scheduler.step()

    lr = get_lr(optimizer=optimizer)
    if writer is not None:
        writer.add_scalar("Train/lr", lr, epoch)

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg, temporal_losses.avg


def validate(model, val_loader, criterion, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()
    temporal_losses = AverageMeter()

    model.eval()
    for batch_idx, (
            speaker_audio_clip,
            _,
            speaker_emotion_clip,
            speaker_3dmm_clip,
            _,
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            _,
    ) in enumerate(tqdm(val_loader)):
        batch_size = speaker_audio_clip.shape[0]
        (speaker_audio_clip,
         speaker_emotion_clip,
         speaker_3dmm_clip,
         listener_emotion_clip,
         listener_3dmm_clip,
         listener_3dmm_clip_personal) = \
            (speaker_audio_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device))

        with torch.no_grad():
            output_prior, output_decoder = model(
                speaker_audio=speaker_audio_clip,
                speaker_emotion_input=speaker_emotion_clip,
                speaker_3dmm_input=speaker_3dmm_clip,
                listener_emotion_input=listener_emotion_clip,
                listener_3dmm_input=listener_3dmm_clip,
                listener_personal_input=listener_3dmm_clip_personal,
            )

            output = criterion(output_prior, output_decoder)
            loss = output["loss"]
            temporal_loss = output["temporal_loss"]
            diff_prior_loss = output["encoded"]
            diff_decoder_loss = output["decoded"]

            whole_losses.update(loss.data.item(), batch_size)
            diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
            diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)
            temporal_losses.update(temporal_loss.data.item(), batch_size)

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg, temporal_losses.avg


def main(args):
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    init_seed(seed=cfg.trainer.seed)  # seed initialization

    # logging
    logging_path = get_logging_path(cfg.trainer.log_dir)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    if cfg.writer:
        writer_path = get_tensorboard_path(cfg.trainer.tb_dir)
        writer = SummaryWriter(writer_path)
    else:
        writer = None

    data_loader = get_dataloader(cfg.dataset)

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    model.to(device)

    if cfg.optimizer.type == "adamW":
        optimizer = optim.AdamW(model.parameters(), betas=cfg.optimizer.args.beta, lr=cfg.optimizer.args.lr,
                                weight_decay=cfg.optimizer.args.weight_decay)
    elif cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), cfg.optimizer.args.lr, weight_decay=cfg.optimizer.args.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), cfg.optimizer.args.lr, momentum=cfg.optimizer.args.momentum,
                              weight_decay=cfg.optimizer.args.weight_decay)
    else:
        NotImplemented("The optimizer {} not implemented.".format(cfg.optimizer.type))

    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)

    if cfg.optimizer.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_loader), eta_min=0,
                                                               last_epoch=-1)
    else:
        scheduler = None

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):
        train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss = (
            train(cfg, model, data_loader, optimizer, scheduler, criterion, epoch, writer, device))

        logging.info(
            "Epoch: {} train_whole_loss: {:.5f} diff_prior_loss: {:.5f} diff_decoder_loss: {:.5f} temporal_loss: {:.5f}"
            .format(epoch + 1, train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss))

        if (epoch+1) % cfg.trainer.val_period == 0:
            save_checkpoint(cfg, model, optimizer)


if __name__ == '__main__':
    main(args=parse_arg())
