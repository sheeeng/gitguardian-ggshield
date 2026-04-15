from functools import wraps
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from ggshield.core.errors import ServiceUnavailableError, handle_exception


T = TypeVar("T")
P = ParamSpec("P")


def exception_wrapper(func: Callable[P, int]) -> Callable[P, int]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> int:
        try:
            return func(*args, **kwargs)
        except Exception as error:
            return handle_exception(error)

    return wrapper


def non_blocking_server_error(func: Callable[P, int]) -> Callable[P, int]:
    """Decorator for git hook/CI commands that should not block when the
    GitGuardian server is unavailable.

    Catches ServiceUnavailableError and returns exit code 0 with a warning,
    instead of blocking the git operation.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> int:
        try:
            return func(*args, **kwargs)
        except ServiceUnavailableError as exc:
            # Lazy import to avoid circular imports (same pattern as handle_exception)
            from ggshield.core import ui

            ui.display_warning(str(exc))
            ui.display_warning("Skipping ggshield checks.")
            return 0

    return wrapper
