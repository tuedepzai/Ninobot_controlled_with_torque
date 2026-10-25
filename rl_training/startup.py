"""Bounded startup recovery, only before a navigation goal is submitted."""

from episode_outcome import InfrastructureFault


class StartupFault(InfrastructureFault):
    pass


def retry_startup(operation, attempts, report):
    if attempts < 1:
        raise ValueError('startup_attempts must be positive')
    for attempt in range(1, attempts+1):
        try:
            return operation()
        except StartupFault as exc:
            report(attempt, attempts, str(exc))
            if attempt == attempts:
                raise
