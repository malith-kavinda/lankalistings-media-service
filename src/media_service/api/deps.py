"""Request-scoped dependencies.

`require_operator` answers one question -- *who is doing this?* -- and every ingestion route needs
it, because an item's audit trail is worthless without an actor.

Two modes. `none` is the local default and trusts a header, which is honest about being no
authentication at all; the startup check refuses it outside local and test, so it cannot reach an
environment where someone might assume otherwise. `static_token` compares a shared secret in
constant time.

The two failure codes are different on purpose (PRD 13.4). 401 says *identify yourself*; 403 says
*you did, and it was not accepted*. Collapsing them would leave an operator with a mistyped token
unable to tell which half of the problem is theirs.
"""

from __future__ import annotations

from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, Request

from media_service.api.errors import ForbiddenError, UnauthenticatedError
from media_service.config import Settings

OPERATOR_HEADER = "X-Operator-Id"
LOCAL_OPERATOR = "local-operator"
BEARER_PREFIX = "Bearer "


def require_operator(request: Request) -> str:
    settings: Settings = request.app.state.settings

    if settings.operator_auth_mode == "none":
        return request.headers.get(OPERATOR_HEADER) or LOCAL_OPERATOR

    header = request.headers.get("Authorization", "")
    if not header.startswith(BEARER_PREFIX):
        raise UnauthenticatedError("Provide an operator token as `Authorization: Bearer <token>`.")

    presented = header.removeprefix(BEARER_PREFIX).strip()
    expected = settings.operator_api_token or ""
    # Constant time: a comparison that returns early leaks the token one character at a time.
    if not compare_digest(presented, expected):
        raise ForbiddenError("The operator token was not recognised.")

    return request.headers.get(OPERATOR_HEADER) or "operator"


Operator = Annotated[str, Depends(require_operator)]
