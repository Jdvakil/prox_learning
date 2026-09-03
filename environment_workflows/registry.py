"""Explicit registry of supported environment lifecycle workflows."""

from __future__ import annotations

from environment_workflows.profile import EnvironmentProfile

_PROFILES: dict[str, EnvironmentProfile] = {}


def register(profile: EnvironmentProfile) -> EnvironmentProfile:
    if profile.environment_id in _PROFILES:
        raise ValueError(f"duplicate environment profile: {profile.environment_id}")
    _PROFILES[profile.environment_id] = profile
    return profile


def get_profile(environment_id: str) -> EnvironmentProfile:
    try:
        return _PROFILES[environment_id]
    except KeyError as error:
        choices = ", ".join(sorted(_PROFILES)) or "(none)"
        raise KeyError(
            f"unknown environment {environment_id!r}; available: {choices}"
        ) from error


def list_profiles() -> tuple[EnvironmentProfile, ...]:
    return tuple(_PROFILES[key] for key in sorted(_PROFILES))


# Profile imports are lightweight and perform explicit registration.
from environment_workflows.environments import pact_place_v12 as _pact_place_v12  # noqa: E402,F401
from environment_workflows.environments import pact_place_v5 as _pact_place_v5  # noqa: E402,F401
