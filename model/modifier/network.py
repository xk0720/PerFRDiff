import os
import torch
import torch.nn as nn
import model as module_arch
from utils.util import checkpoint_load


def compute_regular_loss(weights):
    loss = 0.0
    for w in weights:
        w = w.reshape(-1, )
        loss = loss + torch.norm(w, 2)
    return loss


class ModifierNetwork(nn.Module):
    def __init__(self, input_dim=512, latent_dim=1024, output_dim=None, num_shared_layers=1):
        super(ModifierNetwork, self).__init__()
        self.shared_layers = nn.ModuleList([nn.Linear(input_dim, latent_dim)
                                            if i == 0 else nn.Linear(latent_dim, latent_dim)
                                            for i in range(num_shared_layers)])
        self.output_dim = output_dim
        self.branches = nn.ModuleList(
            [nn.Linear(latent_dim, torch.prod(output_dim[i])) for i in range(len(output_dim))])

    def forward(self, x):
        for layer in self.shared_layers:
            x = torch.relu(layer(x))

        outputs = [branch(x).view(list(self.output_dim[i])) for i, branch in enumerate(self.branches)]
        return outputs

    def get_model_name(self):
        return self.__class__.__name__


class MainNetUnified(nn.Module):
    def __init__(self, cfg, main_net, device):
        super().__init__()
        self.main_net = main_net
        self.modified_layers = cfg.main_model.args.modified_layers
        num_shared_layers = cfg.main_model.args.num_shared_layers
        input_dim = cfg.main_model.args.get("input_dim", 512)
        latent_dim = cfg.main_model.args.get("latent_dim", 1024)
        embed_dim = cfg.main_model.args.get("embed_dim", 512)

        self.hypernet_predict = cfg.main_model.args.get("predict", "shift")
        self.crossattn_modify = cfg.main_model.args.get("modify", "all")

        def target_modules():
            return {n: p.cuda() for n, p in self.main_net.named_modules()
                    if n in self.modified_layers}

        hooked_modules = target_modules()

        weight_shapes = self.main_net.obtain_shapes(self.modified_layers)

        self.weight_shapes = []
        for layer_name in list(hooked_modules.keys()):
            if 'multihead_attn' in layer_name and self.crossattn_modify == 'kv':
                original_dim = weight_shapes[layer_name][0]
                weight_shape = torch.tensor([2 * torch.div(original_dim, 3, rounding_mode='trunc'),
                                             weight_shapes[layer_name][1]])
                self.weight_shapes.append(weight_shape)
            else:
                self.weight_shapes.append(weight_shapes[layer_name])

        self.hypernet = ModifierNetwork(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=self.weight_shapes,
            num_shared_layers=num_shared_layers,
        )

        self.regularization = cfg.main_model.args.regularization
        self.regular_w = cfg.main_model.args.regular_w

        self.person_encoder = self.main_net.diffusion_decoder.person_encoder
        # self.person_encoder = getattr(module_arch, cfg.person_specific.type)(device, **cfg.person_specific.args)
        # model_path = cfg.person_specific.checkpoint_path
        # assert os.path.exists(model_path), (
        #     "Miss checkpoint for model: {}.".format(model_path))
        # checkpoint = torch.load(model_path, map_location='cpu')
        # state_dict = checkpoint['state_dict']
        # self.person_encoder.load_state_dict(state_dict)
        # print(f'Successfully load checkpoint: {model_path}')

        self.mode = cfg.mode
        if self.mode == "test":  # load checkpoints
            checkpoint_load(cfg, self.hypernet, device, checkpoint_path=None)
            # print(f'Successfully load checkpoint: {self.hypernet.get_model_name()}')

        def original_weights():
            weight_list = []
            for name, module in hooked_modules.items():
                if hasattr(module, 'weight'):
                    weight_list.append(getattr(module, 'weight'))
                elif hasattr(module, 'in_proj_weight'):
                    weight_list.append(getattr(module, 'in_proj_weight'))
                else:
                    raise ValueError("The module has either weight or in_proj_weight attribute.")
            return weight_list

        original_weights = original_weights()
        for name, module in hooked_modules.items():
            if hasattr(module, 'weight'):
                del hooked_modules[name]._parameters['weight']
            elif hasattr(module, 'in_proj_weight'):
                del hooked_modules[name]._parameters['in_proj_weight']
            else:
                raise ValueError("The module has either weight or in_proj_weight attribute.")

        def new_forward():
            for i, name in enumerate(list(hooked_modules.keys())):
                delta_w = self.kernel[i]

                if 'linear' in name or 'to_emotion' in name:
                    if self.hypernet_predict == 'shift':
                        hooked_modules[name].weight = original_weights[i] + delta_w
                    elif self.hypernet_predict == 'offset':
                        hooked_modules[name].weight = original_weights[i] * (self.tensor_1 + delta_w)
                    elif self.hypernet_predict == 'weight':
                        hooked_modules[name].weight = delta_w

                else:
                    if 'multihead_attn' in name and self.crossattn_modify == 'kv':
                        delta_w = torch.cat((self.tensor_0, delta_w), dim=0)

                    if self.hypernet_predict == 'shift':
                        hooked_modules[name].in_proj_weight = original_weights[i] + delta_w
                    elif self.hypernet_predict == 'offset':
                        hooked_modules[name].in_proj_weight = original_weights[i] * (self.tensor_1 + delta_w)
                    elif self.hypernet_predict == 'weight':
                        hooked_modules[name].in_proj_weight = delta_w

        self.new_forward = new_forward
        self.tensor_0 = torch.zeros(size=(embed_dim, embed_dim)).to(device)
        self.tensor_1 = torch.tensor(1.0).to(device)

    def forward(self, x, p):
        _, p = self.person_encoder(p)
        self.kernel = self.hypernet(p)
        self.new_forward()
        output = self.main_net(**x)

        if self.regularization:
            norm_loss = self.regular_w * compute_regular_loss(self.kernel)
            return output, norm_loss
        else:
            return output, torch.tensor(0.0)
            