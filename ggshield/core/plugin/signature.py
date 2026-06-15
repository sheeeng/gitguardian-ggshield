"""
Plugin signature verification using sigstore.

Provides keyless signature verification for plugin wheels using sigstore
bundles. Signatures are identity-based: ggshield trusts a configured set
of GitHub Actions workflow identities (OIDC).

Verification always runs against the trust root bundled inside our pinned
sigstore dependency (see ``_bundled_verifier``). ggshield never refreshes TUF
metadata over the network, because that refresh fails on locked-down / proxied
networks ("failed to refresh TUF metadata"). The bundled root is enough to
verify GitGuardian's own plugin signatures; the trade-off is that trust-root
freshness tracks the pinned sigstore version, so keep that dependency current.

Verification modes (these decide how a result is handled, not how it is computed):
- STRICT: block unsigned or invalid plugins
- WARN: log a warning but allow loading (the ``--allow-unsigned`` override)
- DISABLED: skip verification entirely
"""

import enum
import functools
import logging
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from sigstore._internal.tuf import DEFAULT_TUF_URL
from sigstore.errors import VerificationError
from sigstore.models import Bundle, TrustedRoot
from sigstore.verify import Verifier
from sigstore.verify.policy import AllOf, GitHubWorkflowRepository, OIDCIssuer


logger = logging.getLogger(__name__)


class SignatureVerificationMode(enum.Enum):
    """How strictly to enforce plugin signatures."""

    STRICT = "strict"
    WARN = "warn"
    DISABLED = "disabled"


class SignatureStatus(enum.Enum):
    """Result of signature verification."""

    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"
    SKIPPED = "skipped"


@dataclass
class TrustedIdentity:
    """An OIDC identity trusted to sign plugins."""

    repository: str
    """GitHub repository (e.g. "GitGuardian/satori")."""

    issuer: str
    """OIDC issuer URL."""


@dataclass
class SignatureInfo:
    """Result of verifying a wheel's signature."""

    status: SignatureStatus
    identity: Optional[str] = None
    message: Optional[str] = None


class SignatureVerificationError(Exception):
    """Raised when signature verification fails in strict mode."""

    def __init__(self, status: SignatureStatus, message: str) -> None:
        self.status = status
        super().__init__(message)


# Default trusted identities: GitHub Actions workflows that are authorized
# to sign plugin wheels.
DEFAULT_TRUSTED_IDENTITIES: List[TrustedIdentity] = [
    TrustedIdentity(
        repository="GitGuardian/satori",
        issuer="https://token.actions.githubusercontent.com",
    ),
]


def get_bundle_path(wheel_path: Path) -> Optional[Path]:
    """Return the sigstore bundle path for a wheel, or None if not found.

    Checks for `.sigstore` first, then `.sigstore.json` (sigstore-python 3.x
    default) so that both old and new naming conventions are supported.
    """
    for ext in (".sigstore", ".sigstore.json"):
        p = wheel_path.parent / (wheel_path.name + ext)
        if p.exists():
            return p
    return None


@functools.lru_cache(maxsize=1)
def _bundled_verifier() -> Verifier:
    """Build a Verifier pinned to the trust root bundled in our sigstore release.

    We deliberately do NOT use ``Verifier.production(offline=True)``: in offline
    mode sigstore reads its shared on-disk TUF target cache and only seeds the
    embedded root when that cache is *absent* (sigstore ``_internal/tuf.py``
    ``TrustUpdater``), so a stale root left by earlier sigstore use on the
    machine would be used instead. We always verify against the root shipped in
    the pinned sigstore distribution: deterministic, and it never touches the
    network (the network refresh is what fails on locked-down / proxied networks
    with "failed to refresh TUF metadata").

    Resolves the same embedded resource sigstore's own ``read_embedded`` reads
    (``sigstore._store/<quoted-tuf-url>/trusted_root.json``); the dedicated unit
    test fails loudly if a sigstore upgrade relocates it.
    """
    embedded = (
        files("sigstore._store") / quote(DEFAULT_TUF_URL, safe="") / "trusted_root.json"
    )
    with as_file(embedded) as trusted_root_path:
        trusted_root = TrustedRoot.from_file(str(trusted_root_path))
    return Verifier(trusted_root=trusted_root)


def verify_wheel_signature(
    wheel_path: Path,
    mode: SignatureVerificationMode,
    trusted_identities: Optional[List[TrustedIdentity]] = None,
) -> SignatureInfo:
    """
    Verify the sigstore signature of a plugin wheel.

    Args:
        wheel_path: Path to the .whl file.
        mode: Verification strictness.
        trusted_identities: OIDC identities to trust.
            Defaults to DEFAULT_TRUSTED_IDENTITIES.

    Returns:
        SignatureInfo with the verification result.

    Raises:
        SignatureVerificationError: In STRICT mode when the signature is
            missing or invalid.
    """
    if mode == SignatureVerificationMode.DISABLED:
        return SignatureInfo(status=SignatureStatus.SKIPPED)

    if trusted_identities is None:
        trusted_identities = DEFAULT_TRUSTED_IDENTITIES

    bundle_path = get_bundle_path(wheel_path)

    # Missing bundle
    if bundle_path is None:
        msg = f"No signature bundle found for {wheel_path.name}"
        if mode == SignatureVerificationMode.STRICT:
            raise SignatureVerificationError(SignatureStatus.MISSING, msg)
        logger.warning("%s", msg)
        return SignatureInfo(status=SignatureStatus.MISSING, message=msg)

    bundle = Bundle.from_json(bundle_path.read_bytes())
    wheel_bytes = wheel_path.read_bytes()
    verifier = _bundled_verifier()

    for trusted in trusted_identities:
        policy = AllOf(
            [
                OIDCIssuer(trusted.issuer),
                GitHubWorkflowRepository(trusted.repository),
            ]
        )
        try:
            verifier.verify_artifact(
                input_=wheel_bytes,
                bundle=bundle,
                policy=policy,
            )
            logger.info(
                "Signature valid for %s (repository: %s)",
                wheel_path.name,
                trusted.repository,
            )
            return SignatureInfo(
                status=SignatureStatus.VALID,
                identity=trusted.repository,
            )
        except VerificationError as e:
            logger.debug(
                "Identity %s did not match for %s: %s",
                trusted.repository,
                wheel_path.name,
                e,
            )
            continue

    # No identity matched
    msg = f"Signature verification failed for {wheel_path.name}: no trusted identity matched"
    if mode == SignatureVerificationMode.STRICT:
        raise SignatureVerificationError(SignatureStatus.INVALID, msg)
    logger.warning("%s", msg)
    return SignatureInfo(status=SignatureStatus.INVALID, message=msg)
