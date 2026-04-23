from functools import wraps
from typing import Callable, TypeVar

import click
from typing_extensions import ParamSpec

from ggshield.cmd.utils.common_options import create_config_callback
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


fail_on_server_error_option = click.option(
    "--fail-on-server-error/--no-fail-on-server-error",
    "fail_on_server_error",
    is_flag=True,
    default=None,
    envvar="GITGUARDIAN_FAIL_ON_SERVER_ERROR",
    help=(
        "Whether git hook and CI scan commands should fail when the GitGuardian"
        " server is unreachable or returns a 5xx response. When disabled, the"
        " command exits with code 0 and a warning is displayed instead of"
        " blocking the git operation. Defaults to enabled. Can also be set with"
        " the `GITGUARDIAN_FAIL_ON_SERVER_ERROR` environment variable."
    ),
    callback=create_config_callback("secret", "fail_on_server_error"),
)


def non_blocking_on_server_error(func: Callable[P, int]) -> Callable[P, int]:
    """Decorator for git hook / CI commands that may opt in to not blocking
    when the GitGuardian server is unavailable.

    Bundles the ``--fail-on-server-error`` CLI option with a handler that
    catches ``ServiceUnavailableError``: when ``secret.fail_on_server_error``
    is False, the command exits with code 0 and a warning; otherwise the error
    is re-raised and handled like any other error.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> int:
        try:
            return func(*args, **kwargs)
        except ServiceUnavailableError as exc:
            # Lazy import to avoid circular imports (same pattern as handle_exception).
            from ggshield.cmd.utils.context_obj import ContextObj
            from ggshield.core import ui

            ctx = click.get_current_context(silent=True)
            fail_on_server_error = True
            if ctx is not None and ctx.obj is not None:
                fail_on_server_error = ContextObj.get(
                    ctx
                ).config.user_config.secret.fail_on_server_error

            if fail_on_server_error:
                raise

            ui.display_error(str(exc))
            ui.display_error("Skipping ggshield checks.")
            return 0

    return fail_on_server_error_option(wrapper)
