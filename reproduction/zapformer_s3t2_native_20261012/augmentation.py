"""Keep native augmentation unchanged; replay its CPU draws to warp CE labels."""
import torch


def augment(implementation, features, lengths, targets=None):
    if targets is None:
        return implementation(features, lengths), None
    # Native time_warp uses its private CPU RNG; masking uses the CUDA RNG.
    if features.device.type != 'cuda':
        raise ValueError('CE augmentation requires CUDA, separating native warp and masking generators')
    generator = implementation.get_generator(torch.device('cpu'))
    replay = torch.Generator(device='cpu')
    replay.set_state(generator.get_state())
    factor = implementation.time_warp_factor
    warped_targets = targets.clone()
    for b, n_tensor in enumerate(lengths):
        n = int(n_tensor)
        if implementation.time_warp_p <= 0 or factor is None or factor < 1:
            continue
        if torch.rand(1, generator=replay).item() > implementation.time_warp_p:
            continue
        if n - factor <= factor + 1:
            continue
        center = torch.randint(factor + 1, n - factor, (), generator=replay).item()
        warped = torch.randint(center - factor, center + factor + 1, (), generator=replay).item()
        if center == warped:
            continue
        # Nearest labels on the same piecewise align_corners=False time transform
        # used by native bicubic interpolation for continuous acoustic features.
        left = torch.floor((torch.arange(warped, device=features.device) + .5) * center / warped).long().clamp(0, center - 1)
        right = torch.floor((torch.arange(n - warped, device=features.device) + .5) * (n - center) / (n - warped)).long().clamp(0, n - center - 1) + center
        warped_targets[b, :n] = targets[b, torch.cat([left, right])]
    result = implementation(features, lengths)
    assert torch.equal(replay.get_state(), generator.get_state()), 'Native warp RNG and CE label replay diverged'
    return result, warped_targets
