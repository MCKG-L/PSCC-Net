import copy
import math
from pathlib import Path
import re

import torch


def _plain_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _model_state(checkpoint):
    state = checkpoint.get('model', checkpoint)
    return {key.removeprefix('module.'): value for key, value in state.items()}


def _validate_state(model, state, path):
    expected = _plain_model(model).state_dict()
    if expected.keys() != state.keys():
        missing = sorted(expected.keys() - state.keys())
        extra = sorted(state.keys() - expected.keys())
        raise RuntimeError(f'Incompatible weights in {path}: missing={missing}, unexpected={extra}')
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise RuntimeError(f'Incompatible tensor {key} in {path}')


def load_model_weights(model, path):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = _model_state(checkpoint)
    _validate_state(model, state, path)
    _plain_model(model).load_state_dict(state)


def _validate_adam_state(optimizer, saved_state, epoch):
    """Optimizer loading checks group counts but not the shapes of Adam moments."""
    if not isinstance(optimizer, torch.optim.Adam):
        return
    entries = saved_state.get('state')
    if not isinstance(entries, dict):
        raise RuntimeError(f'Missing Adam state at epoch {epoch}')
    for saved_group, current_group in zip(saved_state['param_groups'], optimizer.param_groups):
        for parameter_id, parameter in zip(saved_group['params'], current_group['params']):
            state = entries.get(parameter_id, {})
            if not state:
                continue  # Adam initializes state lazily for parameters with gradients.
            required = ['exp_avg', 'exp_avg_sq']
            if saved_group.get('amsgrad', False):
                required.append('max_exp_avg_sq')
            for key in required:
                value = state.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise RuntimeError(f'Incompatible Adam {key} for parameter {parameter_id} at epoch {epoch}')
            step = state.get('step')
            if isinstance(step, torch.Tensor) and step.numel() == 1:
                step = step.item()
            if not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0:
                raise RuntimeError(f'Invalid Adam step for parameter {parameter_id} at epoch {epoch}')


def resume_training(models, optimizer, checkpoint_root):
    """Restore a complete legacy three-file checkpoint, or fail without mutation."""
    root = Path(checkpoint_root)
    epochs = []
    for name in models:
        pattern = re.compile(rf'{re.escape(name)}_(\d+)\.pth')
        for path in (root / f'{name}_checkpoint').glob('*.pth'):
            match = pattern.fullmatch(path.name)
            if match:
                epochs.append(int(match.group(1)))
    if not epochs:
        return 0
    epoch = max(epochs)
    states = {}
    checkpoints = {}
    for name, model in models.items():
        path = root / f'{name}_checkpoint' / f'{name}_{epoch}.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Incomplete checkpoint at epoch {epoch}: missing {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = _model_state(checkpoint)
        _validate_state(model, state, path)
        states[name] = state
        checkpoints[name] = checkpoint
    optimizer_state = checkpoints['DetectionHead'].get('optimizer')
    if optimizer_state is None:
        raise RuntimeError(f'Missing optimizer in DetectionHead checkpoint at epoch {epoch}')
    saved_groups = optimizer_state.get('param_groups', [])
    if (len(saved_groups) != len(optimizer.param_groups) or
            any(len(saved['params']) != len(current['params'])
                for saved, current in zip(saved_groups, optimizer.param_groups))):
        raise RuntimeError(f'Incompatible optimizer at epoch {epoch}')
    _validate_adam_state(optimizer, optimizer_state, epoch)

    originals = {name: copy.deepcopy(_plain_model(model).state_dict())
                 for name, model in models.items()}
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    try:
        for name, model in models.items():
            _plain_model(model).load_state_dict(states[name])
        optimizer.load_state_dict(optimizer_state)
    except Exception as error:
        for name, model in models.items():
            _plain_model(model).load_state_dict(originals[name])
        optimizer.load_state_dict(original_optimizer)
        raise RuntimeError(f'Cannot resume epoch {epoch}; original states restored') from error
    return epoch
