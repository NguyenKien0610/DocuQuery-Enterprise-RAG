import hmac
import os
import re
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException

WORKSPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class RequestContext:
    workspace_id: str


def require_request_context(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
) -> RequestContext:
    configured_key = os.getenv("DOCUQUERY_API_KEY", "")
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )
    if x_api_key is None or not hmac.compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid API credentials.")
    if x_workspace_id is None or WORKSPACE_PATTERN.fullmatch(x_workspace_id) is None:
        raise HTTPException(status_code=422, detail="Invalid workspace ID.")
    return RequestContext(workspace_id=x_workspace_id)


RequestContextDep = Annotated[RequestContext, Depends(require_request_context)]
