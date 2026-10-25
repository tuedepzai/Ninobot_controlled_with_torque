
# Nino Robotics Development Guidelines

## Environment
- ROS 2 Jazzy
- Gazebo Harmonic
- Ubuntu WSL2 + Docker Compose
- NVIDIA RTX 5060

## Development Rules
- Do not modify Docker GPU or WSLg settings unless requested.
- Keep ROS packages inside src/.
- Keep RL experiments inside rl_training/.
- Do not modify robot URDF or controller parameters without explaining why.
- Never delete training results or Git history without permission.
- Prefer small, reviewable changes.
- Run tests after modifications when possible.

## Learning Mode
- Explain new ROS 2 and RL concepts before implementation.
- Give the student opportunities to implement core algorithms.
- Explain mathematical equations and physical assumptions.
- Prioritize understanding over generating large amounts of code.
