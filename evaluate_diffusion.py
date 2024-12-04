import torch
import argparse
from tqdm import tqdm
from metric import *
from dataset.dataset import get_dataloader
from utils.util import load_config, init_seed, get_logging_path
import model as module_arch


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--config", type=str, help="config path", default='./configs/diffusion_model.yaml')
    args = parser.parse_args()
    return args


def evaluate(cfg, device, model, data_loader):
    model.eval()

    speaker_emotion_list = []
    listener_emotion_gt_list = []
    listener_emotion_pred_list = []

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
    ) in enumerate(tqdm(data_loader)):
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

        listener_emotion_gt = listener_emotion_clip.detach().clone().cpu()
        listener_emotion_clip = listener_emotion_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)
        listener_3dmm_clip = listener_3dmm_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)

        with torch.no_grad():
            _, listener_emotion_pred = model(
                speaker_audio=speaker_audio_clip,
                speaker_emotion_input=speaker_emotion_clip,
                speaker_3dmm_input=speaker_3dmm_clip,
                listener_emotion_input=listener_emotion_clip,
                listener_3dmm_input=listener_3dmm_clip,
                listener_personal_input=listener_3dmm_clip_personal,
            )
            listener_emotion_pred = listener_emotion_pred["prediction_emotion"]

            speaker_emotion_list.append(speaker_emotion_clip.detach().cpu())
            listener_emotion_pred_list.append(listener_emotion_pred.detach().cpu())
            listener_emotion_gt_list.append(listener_emotion_gt.detach().cpu())

    all_speaker_emotion = torch.cat(speaker_emotion_list, dim=0)
    all_listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim=0)
    all_listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim=0)

    print("-----------------Evaluating Metric-----------------")
    p = cfg.test_dataset.threads
    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC_mp(all_listener_emotion_pred, all_speaker_emotion, p=p)
    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_mp(cfg.test_dataset, all_listener_emotion_pred, all_listener_emotion_gt, val_test='test', p=p)
    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_mp(cfg.test_dataset, all_listener_emotion_pred, all_listener_emotion_gt, val_test='test', p=p)
    FRDvs = compute_FRDvs(all_listener_emotion_pred)
    FRVar = compute_FRVar(all_listener_emotion_pred)
    smse = compute_s_mse(all_listener_emotion_pred)

    return FRC, FRD, FRDvs, FRVar, smse, TLCC


def main(args):
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    init_seed(seed=cfg.trainer.seed)  # seed initialization

    data_loader = get_dataloader(cfg.test_dataset)
    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')
        # render = Render('cuda')
    else:
        device = torch.device('cpu')

    model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    model.to(device)

    FRC, FRD, FRDvs, FRVar, smse, TLCC = evaluate(cfg, device, model, data_loader)
    print("FRC: {:.5f}  FRD: {:.5f}  FRDvs: {:.5f}  FRVar: {:.5f}  smse: {:.5f}  TLCC: {:.5f}"
          .format(FRC, FRD, FRDvs, FRVar, smse, TLCC))


if __name__ == '__main__':
    main(args=parse_arg())
