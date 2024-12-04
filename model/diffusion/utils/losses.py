from __future__ import print_function

import torch


def TemporalLoss(Y):
    diff = Y[:, 1:, :] - Y[:, :-1, :]
    t_loss = torch.mean(torch.norm(diff, dim=2, p=2) ** 2)
    return t_loss


def L1Loss(prediction, target, k=1, reduction="min", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    loss = (torch.abs(prediction - target)).mean(axis=-1)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELoss(prediction, target, k=1, reduction="mean", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    loss = ((prediction - target) ** 2).mean(axis=-1)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELossWithAct(prediction, target, k=1, reduction="mean", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    bs, k, _ = prediction.shape
    prediction = prediction.reshape(bs, k, 50, 25)

    AU = prediction[:, :, :, :15]
    AU = torch.sigmoid(AU)

    middle_feat = prediction[:, :, :, 15:17]
    middle_feat = torch.tanh(middle_feat)

    emotion = prediction[:, :, :, 17:]
    emotion = torch.softmax(emotion, dim=-1)

    prediction = torch.cat((AU, middle_feat, emotion), dim=-1)
    prediction = prediction.reshape(bs, k, -1)
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"

    loss = ((prediction - target) ** 2).mean(axis=-1)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def KApproMSELoss(prediction, target, k, **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    bs, _, feat_dim = prediction.shape
    metrics = torch.zeros(size=(bs, 0, k)).to(prediction.device)
    preds = prediction.detach().clone()
    for idk in range(k):
        pred = preds[:, idk:idk + 1, :]
        pred = pred.repeat(1, k, 1)
        mse = ((pred - target) ** 2).mean(axis=-1).unsqueeze(1)
        metrics = torch.cat((metrics, mse), dim=1)
    minimum_mse = torch.argmin(metrics, dim=-1, keepdim=True)
    minimum_mse = minimum_mse.repeat(1, 1, feat_dim).long()
    new_target = torch.gather(target, 1, minimum_mse)
    loss = MSELoss(prediction, new_target, k=1, reduction="mean")

    return loss


def DiffusionLoss(
        output_prior,
        output_decoder,
        losses_type=['MSELoss', 'MSELoss'],
        losses_multipliers=[1, 1],
        losses_decoded=[False, True],
        k=10,
        temporal_loss_w=0.1,
        **kwargs):
    encoded_prediction = output_prior["encoded_prediction"].squeeze(-2)
    encoded_target = output_prior["encoded_target"].squeeze(-2)
    prediction_emotion = output_decoder["prediction_emotion"]
    target_emotion = output_decoder["target_emotion"]

    _, _, window_size, emotion_dim = prediction_emotion.shape
    losses_dict = {"loss": 0.0}

    losses_dict["temporal_loss"] = TemporalLoss(prediction_emotion.reshape(-1, window_size, emotion_dim))
    losses_dict["loss"] += losses_dict["temporal_loss"] * temporal_loss_w
    assert temporal_loss_w <= 0.0, "we first disregard temporal loss."

    prediction_emotion = prediction_emotion.reshape(-1, k, window_size * emotion_dim)
    target_emotion = target_emotion.reshape(-1, k, window_size * emotion_dim)

    for loss_name, w, decoded in zip(losses_type, losses_multipliers, losses_decoded):
        loss_final_name = f"{'decoded' if decoded else 'encoded'}"

        if decoded:
            losses_dict[loss_final_name] = eval(loss_name)(prediction_emotion, target_emotion, k=k)
        else:
            losses_dict[loss_final_name] = eval(loss_name)(encoded_prediction, encoded_target, k=k)

        losses_dict["loss"] += losses_dict[loss_final_name] * w

    return losses_dict
