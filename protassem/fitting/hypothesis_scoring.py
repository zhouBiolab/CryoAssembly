
import torch
from pareconv.modules.ops import apply_transform


def select_best_hypothesis(ref_corr_points, src_corr_points, transforms,
                           acceptance_radius, chunk_size=0):
    num_hypotheses = int(transforms.shape[0])
    if num_hypotheses == 0:
        raise ValueError('transforms is empty: no hypotheses to score')

    if chunk_size <= 0 or chunk_size >= num_hypotheses:
        aligned = apply_transform(src_corr_points.unsqueeze(0), transforms)
        residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - aligned, dim=2)
        masks = torch.lt(residuals, acceptance_radius)
        best_index = int(masks.sum(dim=1).argmax())
        return best_index, masks[best_index]

    best_index = 0
    best_count = None
    best_mask = None
    for start in range(0, num_hypotheses, chunk_size):
        stop = min(start + chunk_size, num_hypotheses)
        chunk = transforms[start:stop]
        aligned = apply_transform(src_corr_points.unsqueeze(0), chunk)
        residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - aligned, dim=2)
        masks = torch.lt(residuals, acceptance_radius)
        counts = masks.sum(dim=1)
        local_index = int(counts.argmax())
        local_count = counts[local_index]
        if best_count is None or bool(local_count > best_count):
            best_count = local_count
            best_index = start + local_index
            best_mask = masks[local_index]
    return best_index, best_mask
