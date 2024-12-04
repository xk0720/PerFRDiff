from tqdm import tqdm
import os
import numpy as np
import torch
from datetime import datetime
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.backends import cudnn


def init_seed(seed, rank=0):
    process_seed = seed + rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    np.random.seed(process_seed)
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_config_from_file(path):
    return OmegaConf.load(path)


def load_config(args=None, config_path=None):
    if args is not None:
        config_from_args = OmegaConf.create(vars(args))
    else:
        config_from_args = OmegaConf.from_cli()
    # config_from_file = OmegaConf.load(cli_conf.pop('config') if config_path is None else config_path)
    config_from_file = load_config_from_file(config_path)
    return OmegaConf.merge(config_from_file, config_from_args)


def store_config(config):
    dir = config.trainer.out_dir
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(config), f)


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(0, 2, 3, 1)


def torch_img_to_np2(img):
    img = img.detach().cpu().numpy()
    img = img * np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img + np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img.transpose(0, 2, 3, 1)
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]

    return img


def _fix_image(image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]
    return image


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collect_grad_value_(parameters):
    grad_values = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        grad_values.append(p.grad.data.abs().mean().item())
    grad_values = np.array(grad_values)
    return grad_values


def get_tensorboard_path(tb_dir):
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = '-'.join(time_str.split(' '))
    time_str = time_str.split('.')[0]
    tb_dir = os.path.join(tb_dir, time_str)
    os.makedirs(tb_dir, exist_ok=True)
    return tb_dir


def get_logging_path(log_dir):
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = '-'.join(time_str.split(' '))
    time_str = time_str.split('.')[0]
    lod_dir = os.path.join(log_dir, time_str)
    return lod_dir


def tsne_visualisation(cfg, listener_features, listeners_label, epoch, split, exp):
    unique_values = torch.unique(listeners_label)
    listener_features = listener_features.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    tsne_results = tsne.fit_transform(listener_features)

    colors = plt.cm.get_cmap('tab10', len(unique_values))
    color_list = [colors(i) for i in range(len(unique_values))]

    # legend put lower center
    plt.figure(figsize=(10, 6))
    for i, value in enumerate(unique_values):
        x = tsne_results[listeners_label == value][:, 0]
        y = tsne_results[listeners_label == value][:, 1]
        plt.scatter(x, y, color=color_list[i], label="Listener {}'s Facial Behaviours".format(i+1))

    plt.title('t-SNE Visualisation of Listener Facial Behaviours', fontsize=15, fontweight='bold')
    plt.legend(fontsize=13, loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=2)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    save_dir_1 = os.path.join(cfg.similarity_outdir, 't-sne', 'exp_' + str(exp), split, 'transparent')
    os.makedirs(save_dir_1, exist_ok=True)
    save_path_1 = os.path.join(save_dir_1, 'lower_center.png')
    plt.savefig(save_path_1, dpi=300, transparent=True)

    save_dir_2 = os.path.join(cfg.similarity_outdir, 't-sne', 'exp_' + str(exp), split, 'normal')
    os.makedirs(save_dir_2, exist_ok=True)
    save_path_2 = os.path.join(save_dir_2, 'lower_center.png')
    plt.savefig(save_path_2, dpi=300)

    plt.close()

    # legend put upper right
    plt.figure()
    plt.figure(figsize=(10, 6))
    for i, value in enumerate(unique_values):
        x = tsne_results[listeners_label == value][:, 0]
        y = tsne_results[listeners_label == value][:, 1]
        plt.scatter(x, y, color=color_list[i], label="Listener {}'s \nFacial Behaviours".format(i + 1))

    plt.title('t-SNE Visualisation of Listener Facial Behaviours', fontsize=15, fontweight='bold')
    plt.legend(fontsize=13, loc='upper right', bbox_to_anchor=(1.34, 1.02), ncol=1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    save_dir_1 = os.path.join(cfg.similarity_outdir, 't-sne', 'exp_' + str(exp), split, 'transparent')
    os.makedirs(save_dir_1, exist_ok=True)
    save_path_1 = os.path.join(save_dir_1, 'upper_right.png')
    plt.savefig(save_path_1, dpi=300, transparent=True)

    save_dir_2 = os.path.join(cfg.similarity_outdir, 't-sne', 'exp_' + str(exp), split, 'normal')
    os.makedirs(save_dir_2, exist_ok=True)
    save_path_2 = os.path.join(save_dir_2, 'upper_right.png')
    plt.savefig(save_path_2, dpi=300)

    plt.close()


def save_checkpoint_pretrain(cfg, model, optimizer):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dir = os.path.join(cfg.trainer.checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f'Successfully save checkpoint: {save_path}')


def save_checkpoint(cfg, model, optimizer):
    diffusion_prior_model = model.diffusion_prior.model
    diffusion_prior_name = diffusion_prior_model.get_model_name()
    diffusion_prior_dir = os.path.join(cfg.trainer.checkpoint_dir, diffusion_prior_name)

    prior_checkpoint = {
        'state_dict': diffusion_prior_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dir = os.path.join(diffusion_prior_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(prior_checkpoint, save_path)
    print(f'Successfully save checkpoint: {save_path}')

    # diffusion decoder net
    diffusion_decoder_model = model.diffusion_decoder.model
    diffusion_decoder_name = diffusion_decoder_model.get_model_name()
    diffusion_decoder_dir = os.path.join(cfg.trainer.checkpoint_dir, diffusion_decoder_name)

    decoder_checkpoint = {
        'state_dict': diffusion_decoder_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dir = os.path.join(diffusion_decoder_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(decoder_checkpoint, save_path)
    print(f'Successfully save checkpoint: {save_path}')


def checkpoint_load(cfg, model, device, checkpoint_path=None):
    if checkpoint_path is None:
        model_name = model.get_model_name()
        checkpoint_dir = cfg.trainer.checkpoint_dir
        checkpoint_path = os.path.join(checkpoint_dir, model_name, 'checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    print(f'Successfully load checkpoint: {checkpoint_path}')


# def checkpoint_resume(cfg, model, device):
#     model_name = model.get_model_name()
#     model_path = os.path.join(cfg.trainer.checkpoint_dir, model_name, cfg.trainer.resume)
#     checkpoint = torch.load(model_path, map_location='cpu')
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device)
#     print(f'Successfully resume checkpoint: {model_path}')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def collect_grad_stats(parameters):
    grad_values = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        # Store the absolute values of gradients
        grad_values.extend(p.grad.data.abs().view(-1).cpu().numpy())

    # Convert to a numpy array for statistical computation
    grad_values = np.array(grad_values)
    if grad_values.size == 0:
        return {"min": None, "max": None, "mean": None}

    # Compute min, max, and mean
    grad_stats = {
        "min": grad_values.min(),
        "max": grad_values.max(),
        "mean": grad_values.mean()
    }
    return grad_stats
