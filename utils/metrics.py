import torch

# Source for Precision/Recall/F1 measurements: https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure

# Computes accuracy weighted by patches (metric used on Kaggle for evaluation)
def patch_accuracy_fn(y_hat: torch.Tensor, y: torch.Tensor, patch_size: int = 16, cutoff: float = 0.25):
    h_patches = y.shape[-2] // patch_size
    w_patches = y.shape[-1] // patch_size
    
    # Only 1 channel for the prediction!
    patches_hat = y_hat.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    patches = y.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    return (patches == patches_hat).to(torch.float32)

# Computes the Precision metric for a batch of predictions (how many retrieved items are relevant)
# By convention: if TP, FP, FN are all zero                  => Precision value has to be 1
#                if TP = 0 and at least FP or FN is non-zero => Precision value has to be 0
def precision_fn(y_hat: torch.Tensor, y: torch.Tensor):
    gt_pos = y == 1.0
    gt_neg = y == 0.0

    pred_pos = y_hat >= 0.5
    pred_neg = y_hat < 0.5

    true_pos = torch.logical_and(pred_pos, gt_pos).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_pos = torch.logical_and(pred_pos, gt_neg).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_neg = torch.logical_and(pred_neg, gt_pos).sum(dim=(-1, -2, -3), dtype=torch.float32)

    denom = true_pos + false_pos
    precision = torch.where(denom == 0, 0, true_pos / denom)
    full_val = denom + false_neg
    return torch.where(full_val == 0, 1, precision)

# Computes the Recall metric for a batch of predictions (how many relevant items are retrieved)
# By convention: if TP, FP, FN are all zero                  => Recall value has to be 1
#                if TP = 0 and at least FP or FN is non-zero => Recall value has to be 0
def recall_fn(y_hat: torch.Tensor, y: torch.Tensor):
    gt_pos = y == 1.0
    gt_neg = y == 0.0

    pred_pos = y_hat >= 0.5
    pred_neg = y_hat < 0.5

    true_pos = torch.logical_and(pred_pos, gt_pos).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_pos = torch.logical_and(pred_pos, gt_neg).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_neg = torch.logical_and(pred_neg, gt_pos).sum(dim=(-1, -2, -3), dtype=torch.float32)

    denom = true_pos + false_neg
    recall = torch.where(denom == 0, 0, true_pos / denom)
    full_val = denom + false_pos
    return torch.where(full_val == 0, 1, recall)

# Computes the F1 score metric for a batch of predictions
# By convention: if TP, FP, FN are all zero                  => F1 score has to be 1
#                if TP = 0 and at least FP or FN is non-zero => F1 score has to be 0
def f1_fn(y_hat: torch.Tensor, y: torch.Tensor):
    precision = precision_fn(y_hat, y)
    recall = recall_fn(y_hat, y)

    denom = precision + recall
    return torch.where(denom == 0, 0, 2*precision*recall / denom)

# Computes the patch F1 score metric for a batch of predictions
def patch_f1_fn(y_hat: torch.Tensor, y: torch.Tensor, patch_size: int=16, cutoff: float=0.25):
    h_patches = y.shape[-2] // patch_size
    w_patches = y.shape[-1] // patch_size
    patches_hat = y_hat.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    patches = y.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff

    return f1_fn(patches_hat, patches)

# Computes pixel-wise Intersection-over-Union per inference image
def iou_fn(y_hat: torch.Tensor, y: torch.Tensor):
    gt_pos = y == 1.0
    gt_neg = y == 0.0

    pred_pos = y_hat >= 0.5
    pred_neg = y_hat < 0.5

    true_pos = torch.logical_and(pred_pos, gt_pos).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_pos = torch.logical_and(pred_pos, gt_neg).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_neg = torch.logical_and(pred_neg, gt_pos).sum(dim=(-1, -2, -3), dtype=torch.float32)

    denom = true_pos + false_pos + false_neg
    return torch.where(denom == 0, 0, true_pos / denom)
