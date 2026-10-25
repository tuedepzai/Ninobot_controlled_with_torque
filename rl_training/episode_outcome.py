"""Keep infrastructure failures out of the robot-task reward/replay stream."""

INFRASTRUCTURE_ERROR = 65001  # Matches EpisodeManager's NavigateToPose result.


class InfrastructureFault(RuntimeError):
    pass


def navigation_outcome(status, error_code, reason):
    if error_code == INFRASTRUCTURE_ERROR:
        raise InfrastructureFault(f'Navigation data/TF failure: {reason}')
    return 'SUCCEEDED' if status == 4 and error_code == 0 else 'FAILED'
