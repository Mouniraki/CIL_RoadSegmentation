def patch_accuracy_fn(y_hat, y, patch_size=16, cutoff=0.25):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // patch_size
    w_patches = y.shape[-1] // patch_size
    patches_hat = y_hat.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    patches = y.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    return (patches == patches_hat).float().mean()

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()