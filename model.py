import torch
import torch.nn as nn
import timm
import math
import config

class SingleObjectKeypointDetector(nn.Module):
  def __init__(self, num_keypoints=config.NUM_KEYPOINTS):
    super().__init__()
    self.num_keypoints = num_keypoints

    self.backbone = timm.create_model(
        config.MODEL_BACKBONE,
        pretrained=True,
        features_only=True,
    )

    last_chs = self.backbone.feature_info[-1]['num_chs']
    prev_chs = self.backbone.feature_info[-2]['num_chs']

    # Last stage: global semantic context
    self.compress_last = nn.Sequential(
        nn.Conv2d(last_chs, 128, kernel_size=1),
        nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )

    # Previous stage: finer spatial detail
    self.compress_prev = nn.Sequential(
        nn.Conv2d(prev_chs, 64, kernel_size=1),
        nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )

    # 128 + 64 = 192 combined features
    self.head = nn.Sequential(
        nn.Linear(192, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_keypoints * 3)
    )

    nn.init.normal_(self.head[-1].weight, mean=0, std=0.01)
    nn.init.constant_(self.head[-1].bias, 0)

  def forward(self, x):
    features = self.backbone(x)
    feat_last = self.compress_last(features[-1])
    feat_prev = self.compress_prev(features[-2])
    out = self.head(torch.cat([feat_last, feat_prev], dim=1))
    return out.view(out.shape[0], self.num_keypoints, 3)

class WingLossWithVisibility(nn.Module):
  def __init__(self, w=config.WING_LOSS_W, epsilon=config.WING_LOSS_EPSILON, lambda_vis=config.VISIBILITY_LOSS_LAMBDA):
    super().__init__()
    self.w = w
    self.epsilon = epsilon
    self.C = self.w - self.w * math.log(1 + w / epsilon)
    self.lambda_vis = lambda_vis
    self.bce_loss = nn.BCEWithLogitsLoss()

  def forward(self, pred, target, target_vis):
    pred_coords = pred[:, :, :2]
    pred_vis_logits = pred[:, :, 2]

    loss_vis = self.bce_loss(pred_vis_logits, target_vis)

    diff = target - pred_coords
    abs_diff = diff.abs()

    is_small_error = abs_diff < self.w
    loss_small = self.w * torch.log(1 + abs_diff / self.epsilon)
    loss_large = abs_diff - self.C
    raw_coord_loss = torch.where(is_small_error, loss_small, loss_large)

    mask = target_vis.unsqueeze(-1).expand_as(raw_coord_loss)
    masked_coord_loss = raw_coord_loss * mask

    num_visible = mask.sum() + 1e-6
    loss_coord = masked_coord_loss.sum() / num_visible

    total_loss = loss_coord + self.lambda_vis * loss_vis
    return total_loss, loss_coord.item(), loss_vis.item()

class ModelWithNormalization(nn.Module):
    def __init__(self, base_model):
        super(ModelWithNormalization, self).__init__()
        self.base_model = base_model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x_normalized = (x - self.mean) / self.std
        return self.base_model(x_normalized)
