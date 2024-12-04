import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, feature, label, device):
        assert len(feature) == len(label), "The length of features and mask is inconsistent."
        N = feature.shape[0]

        label = label.contiguous().view(-1, 1)  # shape: (N, 1)
        mask = torch.eq(label, label.T).float().to(device)  # shape: (N, N)

        anchor_feature = feature
        contrast_feature = feature
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # shape: (N, N)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(N).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # shape: (N, N)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        return loss.mean()


def MSELoss_AE_v1(prediction, target, target_coefficients, mu, logvar, coefficients_emotion,
                  w_mse=1, w_kld=1, w_coeff=1, **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, seq_length, features]"
    batch_size = prediction.shape[0]

    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    coefficients_emotion = coefficients_emotion.reshape(coefficients_emotion.shape[0], -1)
    target_coefficients = target_coefficients.reshape(target_coefficients.shape[0], -1)

    MSE = ((prediction - target) ** 2).mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    COEFF = ((coefficients_emotion - target_coefficients) ** 2).mean()

    loss_r = w_mse * MSE + w_kld * KLD + w_coeff * COEFF
    return {"loss": loss_r, "mse": MSE, "kld": KLD, "coeff": COEFF}


def MSELoss_AE_v2(prediction, target, target_coefficients, mu, logvar, coefficients_3dmm,
                  w_mse=1, w_kld=1, w_coeff=1, **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, seq_length, features]"
    batch_size = prediction.shape[0]

    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    coefficients_3dmm = coefficients_3dmm.reshape(coefficients_3dmm.shape[0], -1)
    target_coefficients = target_coefficients.reshape(target_coefficients.shape[0], -1)

    MSE = ((prediction - target) ** 2).mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    COEFF = ((coefficients_3dmm - target_coefficients) ** 2).mean()

    loss_r = w_mse * MSE + w_kld * KLD + w_coeff * COEFF
    return {"loss": loss_r, "mse": MSE, "kld": KLD, "coeff": COEFF}


def MSELoss_AE_audio(prediction, target, mu, logvar, w_mse=1, w_kld=1, **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, seq_length, features]"
    batch_size = prediction.shape[0]

    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    MSE = ((prediction - target) ** 2).mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    loss_r = w_mse * MSE + w_kld * KLD
    return {"loss": loss_r, "mse": MSE, "kld": KLD}


# VAE loss for auto-encoding
class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class VAELoss(nn.Module):
    def __init__(self, kl_p=0.0002):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, gt_emotion, gt_3dmm, pred_emotion, pred_3dmm, distribution):
        rec_loss = self.mse(pred_emotion, gt_emotion) + self.mse(pred_3dmm[:, :, :52],
                                                                 gt_3dmm[:, :, :52]) + 10 * self.mse(
            pred_3dmm[:, :, 52:], gt_3dmm[:, :, 52:])

        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss

        return loss, rec_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"


def div_loss(Y_1, Y_2):
    loss = 0.0
    b, t, c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b, 1, -1), Y_2.view(b, 1, -1)], dim=1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist / 100).exp().mean()
    loss /= b
    return loss
