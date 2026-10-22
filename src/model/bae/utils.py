import re

import torch

_RES_BLOCK_RENAME = {'norm1': 'block.0', 'conv1': 'block.2', 'norm2': 'block.3', 'conv2': 'block.5'}


def _rename(key: str) -> str:
    key = key.removeprefix('ae.').replace('generator.', 'decoder.')
    if key.startswith('quantize.'):
        key = 'quantizer.' + key.removeprefix('quantize.')
    return key


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert a checkpoint of the original BinaryGAN to the `BinaryAutoEncoder` layout.

    Mapping:
    - `ae.generator.*` -> `decoder.*`, `ae.quantize.*` -> `quantizer.*`;
    - `ResBlock` norm1/conv1/norm2/conv2 -> `block.{0,2,3,5}`;
    - `ResBlock` conv_out -> `shortcut`, kept only when the number of channels changes
      (the original stores an unused identity-sized conv_out for every block);
    - `AttentionBlock` q/k/v -> `qkv` (concatenated along output channels);
    - discriminator, LPIPS and the ELBO-only `logsigma` head are dropped.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        State dict of the original BinaryGAN.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict of `BinaryAutoEncoder`.
    """
    with_shortcut = {
        _rename(key).removesuffix('.weight')
        for key, value in state_dict.items()
        if key.endswith('.conv_out.weight') and value.shape[0] != value.shape[1]
    }

    converted: dict[str, torch.Tensor] = {}
    qkv: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not key.startswith('ae.') or '.logsigma.' in key:
            continue
        key = _rename(key)
        if m := re.fullmatch(r'(.*blocks\.\d+)\.(q|k|v)\.(weight|bias)', key):
            qkv.setdefault(f'{m[1]}.qkv.{m[3]}', {})[m[2]] = value
        elif m := re.fullmatch(r'(.*blocks\.\d+)\.conv_out\.(weight|bias)', key):
            if f'{m[1]}.conv_out' in with_shortcut:
                converted[f'{m[1]}.shortcut.{m[2]}'] = value
        elif m := re.fullmatch(r'(.*blocks\.\d+)\.(norm1|conv1|norm2|conv2)\.(weight|bias)', key):
            converted[f'{m[1]}.{_RES_BLOCK_RENAME[m[2]]}.{m[3]}'] = value
        else:
            converted[key] = value

    for key, parts in qkv.items():
        converted[key] = torch.cat([parts['q'], parts['k'], parts['v']], dim=0)
    return converted
