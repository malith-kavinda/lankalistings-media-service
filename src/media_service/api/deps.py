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

**What the returned identity is worth.** In both modes it comes from a header the caller controls,
so it is a claim, not an authenticated identity: `static_token` proves the caller holds the one
shared secret, and says nothing about *which* person that is. The identity is still recorded,
because an audit trail naming a reviewer is more useful than one naming nobody, but `approved_by`
carries no non-repudiation weight until operators have individual credentials. That is a real gap
and it belongs in the auth mode, not in the callers -- every one of which treats this string as a
person.

What is enforced here is the shape: bounded length and no control characters. The value reaches a
`String(64)` column, where an over-long one is a 500 rather than a 422, and several log lines, where
an embedded newline lets a caller forge log entries.
"""

from __future__ import annotations

import re
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, Request

from media_service.api.errors import ForbiddenError, UnauthenticatedError
from media_service.config import Settings

OPERATOR_HEADER = "X-Operator-Id"
LOCAL_OPERATOR = "local-operator"
BEARER_PREFIX = "Bearer "

# Matches the `String(64)` columns the value is written to.
MAX_OPERATOR_ID_LENGTH = 64
_ALLOWED_OPERATOR_ID = re.compile(r"^[A-Za-z0-9._@+-]{1,64}$")


def require_operator(request: Request) -> str:
    settings: Settings = request.app.state.settings

    if settings.operator_auth_mode == "none":
        return _claimed_operator(request, default=LOCAL_OPERATOR)

    header = request.headers.get("Authorization", "")
    if not header.startswith(BEARER_PREFIX):
        raise UnauthenticatedError("Provide an operator token as `Authorization: Bearer <token>`.")

    presented = header.removeprefix(BEARER_PREFIX).strip()
    expected = settings.operator_api_token or ""
    # Constant time: a comparison that returns early leaks the token one character at a time.
    if not compare_digest(presented, expected):
        raise ForbiddenError("The operator token was not recognised.")

    return _claimed_operator(request, default="operator")


def _claimed_operator(request: Request, *, default: str) -> str:
    claimed = (request.headers.get(OPERATOR_HEADER) or "").strip()
    if not claimed:
        return default
    if not _ALLOWED_OPERATOR_ID.match(claimed):
        raise ForbiddenError(
            f"{OPERATOR_HEADER} must be at most {MAX_OPERATOR_ID_LENGTH} characters of "
            "letters, digits, and . _ @ + -"
        )
    return claimed


Operator = Annotated[str, Depends(require_operator)]
