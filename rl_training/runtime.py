"""Verify the requested training device before opening Gazebo."""

import torch


def resolve_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cpu':
        return 'cpu', 'CPU'
    if requested != 'cuda':
        raise ValueError(f'Unknown device: {requested}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install the CUDA PyTorch build from rl_training/README.md, '
                           'or select --device cpu explicitly.')
    # is_available() alone does not check whether kernels support the GPU.
    x = torch.ones((16, 16), device='cuda', requires_grad=True)
    (x @ x).sum().backward()
    torch.cuda.synchronize()
    return 'cuda', f'{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}'


def verify_model_device(model, requested):
    """Verify SB3 parameters AND an actor forward pass, without moving the robot."""
    policy = str(next(model.policy.parameters()).device)
    actor = str(next(model.actor.parameters()).device)
    critic = str(next(model.critic.parameters()).device)
    with torch.no_grad():
        observation = torch.zeros((1, *model.observation_space.shape), device=model.device)
        action = model.actor(observation, deterministic=True)
    forward = str(action.device)
    if requested == 'cuda':
        if not all(device.startswith('cuda') for device in (policy, actor, critic, forward)):
            raise RuntimeError(f'SB3 CUDA mismatch: {policy}, {actor}, {critic}, {forward}')
        torch.cuda.synchronize()
    optimizer_devices = sorted({str(value.device)
        for optimizer in (model.actor.optimizer, model.critic.optimizer)
        for state in optimizer.state.values() for key, value in state.items()
        if isinstance(value, torch.Tensor) and key != 'step'})
    return dict(verified=True, requested=requested, policy_device=policy, actor_device=actor,
                critic_device=critic, actor_forward_device=forward,
                optimizer_state_devices=optimizer_devices,
                gpu_name=torch.cuda.get_device_name(0) if requested == 'cuda' else None,
                cuda_version=torch.version.cuda, timesteps=model.num_timesteps,
                gradient_updates=model._n_updates)
