import os
import logging
import torch
import argparse
from torch import optim
from functools import partial
from tqdm import tqdm
from dataset.dataset import get_dataloader
import model.diffusion.utils.losses as module_loss
from torch.utils.tensorboard import SummaryWriter
from utils.util import load_config, init_seed, get_logging_path, get_tensorboard_path, AverageMeter, collect_grad_stats
import model as module_arch


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--writer", type=bool, help="whether use tensorboard", default=True)
    parser.add_argument("--config", type=str, help="config path", default='./configs/rewrite_weight.yaml')
    args = parser.parse_args()
    return args


def save_checkpoint(cfg, net, optimizer):
    checkpoint = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dir = os.path.join(cfg.trainer.checkpoint_dir, net.get_model_name())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f'Successfully save checkpoint: {save_path}')


def train(cfg, model, data_loader, optimizer_hypernet, optimizer_mainnet, criterion, epoch, writer, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()

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

        input_dict = {
            "speaker_audio": speaker_audio_clip,
            "speaker_emotion_input": speaker_emotion_clip,
            "speaker_3dmm_input": speaker_3dmm_clip,
            "listener_emotion_input": listener_emotion_clip,
            "listener_3dmm_input": listener_3dmm_clip,
            "listener_personal_input": listener_3dmm_clip_personal,
        }

        [output_prior, output_decoder], regular_loss = (
            model(x=input_dict, p=listener_3dmm_clip_personal))

        output = criterion(output_prior, output_decoder)

        loss = output["loss"] + regular_loss
        diff_prior_loss = output["encoded"]
        diff_decoder_loss = output["decoded"]

        iteration = batch_idx + len(data_loader) * epoch

        if writer is not None:
            writer.add_scalar("Train/whole_loss", loss.data.item(), iteration)
            writer.add_scalar("Train/diff_prior_loss", diff_prior_loss.data.item(), iteration)
            writer.add_scalar("Train/diff_decoder_loss", diff_decoder_loss.data.item(), iteration)
            writer.add_scalar("Train/regular_loss", regular_loss.data.item(), iteration)

        whole_losses.update(loss.data.item(), batch_size)
        diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
        diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)

        optimizer_mainnet.zero_grad()
        loss.backward()
        optimizer_hypernet.step()

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg


def validate(model, val_loader, criterion, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()

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
            input_dict = {
                "speaker_audio": speaker_audio_clip,
                "speaker_emotion_input": speaker_emotion_clip,
                "speaker_3dmm_input": speaker_3dmm_clip,
                "listener_emotion_input": listener_emotion_clip,
                "listener_3dmm_input": listener_3dmm_clip,
                "listener_personal_input": listener_3dmm_clip_personal,
            }
            [output_prior, output_decoder], regular_loss = model(x=input_dict, p=listener_3dmm_clip_personal)

            output = criterion(output_prior, output_decoder)
            loss = output["loss"] + regular_loss
            diff_prior_loss = output["encoded"]
            diff_decoder_loss = output["decoded"]

            whole_losses.update(loss.data.item(), batch_size)
            diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
            diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg


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

    # diffusion model
    diff_model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    diff_model.to(device)

    main_model = getattr(module_arch, cfg.main_model.type)(cfg, diff_model, device)
    main_model.to(device)

    optimizer_hypernet = optim.SGD(params=main_model.hypernet.parameters(),
                                   lr=cfg.main_model.optimizer_hypernet.args.lr,
                                   momentum=cfg.main_model.optimizer_hypernet.args.momentum,
                                   weight_decay=cfg.main_model.optimizer_hypernet.args.weight_decay)

    optimizer_mainnet = optim.SGD(params=main_model.parameters(),
                                  lr=cfg.main_model.optimizer_mainnet.args.lr,
                                  momentum=cfg.main_model.optimizer_mainnet.args.momentum,
                                  weight_decay=cfg.main_model.optimizer_mainnet.args.weight_decay)

    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):
        train_loss, diff_prior_loss, diff_decoder_loss = train(
            cfg, main_model, data_loader, optimizer_hypernet, optimizer_mainnet, criterion, epoch, writer, device
        )

        logging.info(
            "Epoch: {} train_whole_loss: {:.5f} diff_prior_loss: {:.5f} diff_decoder_loss: {:.5f}"
            .format(epoch + 1, train_loss, diff_prior_loss, diff_decoder_loss))

        if (epoch + 1) % cfg.trainer.val_period == 0:
            save_checkpoint(cfg, main_model.hypernet, optimizer_hypernet)


if __name__ == '__main__':
    main(args=parse_arg())
