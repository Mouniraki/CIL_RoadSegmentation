import torch

# Computes accuracy weighted by patches (metric used on Kaggle for evaluation)
def patch_accuracy_fn(y_hat: torch.Tensor, y: torch.Tensor, patch_size: int = 16, cutoff: float = 0.25):
    h_patches = y.shape[-2] // patch_size
    w_patches = y.shape[-1] // patch_size
    
    # Only 1 channel for the prediction!
    patches_hat = y_hat.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    patches = y.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    return (patches == patches_hat).float().mean().item()

# Computes the Precision metric for a batch of predictions (how many retrieved items are relevant)
def precision_fn(y_hat: torch.Tensor, y: torch.Tensor):
    gt_pos = y == 1.0
    gt_neg = y == 0.0

    pred_pos = y_hat >= 0.5

    true_pos = torch.logical_and(pred_pos, gt_pos).sum(dim=(-1, -2, -3))
    false_pos = torch.logical_and(pred_pos, gt_neg).sum(dim=(-1, -2, -3))
    # print(f"PRECISION_TP:{true_pos}")
    # print(f"PRECISION_FP:{false_pos}")

    return (true_pos / (true_pos + false_pos))

# Computes the Recall metric for a batch of predictions (how many relevant items are retrieved)
def recall_fn(y_hat: torch.Tensor, y: torch.Tensor):
    gt_pos = y == 1.0

    pred_pos = y_hat >= 0.5
    pred_neg = y_hat < 0.5

    true_pos = torch.logical_and(pred_pos, gt_pos).sum(dim=(-1, -2, -3))
    false_neg = torch.logical_and(pred_neg, gt_pos).sum(dim=(-1, -2, -3))
    # print(f"RECALL_TP:{true_pos}")
    # print(f"RECALL_FN:{false_neg}")

    return (true_pos / (true_pos + false_neg))

# Computes the F1 score metric for a batch of predictions
def f1_fn(y_hat: torch.Tensor, y: torch.Tensor):
    precision = precision_fn(y_hat, y)
    recall = recall_fn(y_hat, y)
    return (2 * precision * recall) / (precision + recall)

# Computes pixel-wise Intersection-over-Union per inference image
def iou_fn(y_hat: torch.Tensor, y: torch.Tensor):
    gt_pos = y == 1.0
    gt_neg = y == 0.0

    pred_pos = y_hat >= 0.5
    pred_neg = y_hat < 0.5

    true_pos = torch.logical_and(pred_pos, gt_pos).sum(dim=(-1, -2, -3))
    false_pos = torch.logical_and(pred_pos, gt_neg).sum(dim=(-1, -2, -3))
    false_neg = torch.logical_and(pred_neg, gt_pos).sum(dim=(-1, -2, -3))

    return (true_pos / (true_pos + false_pos + false_neg))
