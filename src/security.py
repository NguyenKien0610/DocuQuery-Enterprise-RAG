import hmac
import json
import os
import re
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from src import state

WORKSPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class RequestContext:
    workspace_id: str
    role: str = "owner"

    def require(self, action: str) -> None:
        allowed = {
            "reader": {"read"},
            "writer": {"read", "write"},
            "owner": {"read", "write", "admin"},
        }
        if action not in allowed.get(self.role, set()):
            raise HTTPException(
                status_code=403, detail="Insufficient workspace permission."
            )


def credentials() -> list[dict]:
    raw = os.getenv("DOCUQUERY_CREDENTIALS")
    if raw:
        entries = json.loads(raw)
        if not isinstance(entries, list) or not entries:
            raise ValueError("Invalid credential configuration")
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("key"), str)
                or not entry["key"]
            ):
                raise ValueError("Invalid credential configuration")
            scopes = entry.get("workspaces")
            if (
                not isinstance(scopes, dict)
                or not scopes
                or any(
                    not WORKSPACE_PATTERN.fullmatch(workspace)
                    or role not in ("reader", "writer", "owner")
                    for workspace, role in scopes.items()
                )
            ):
                raise ValueError("Invalid workspace grant")
        return entries
    key = os.getenv("DOCUQUERY_API_KEY", "")
    if not key:
        raise ValueError("Missing credentials")
    workspace = os.getenv("DOCUQUERY_WORKSPACE_ID", "default")
    if not WORKSPACE_PATTERN.fullmatch(workspace):
        raise ValueError("Invalid default workspace")
    return [{"key": key, "workspaces": {workspace: "owner"}}]


def require_request_context(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
) -> RequestContext:
    try:
        configured = credentials()
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )
    matched = next(
        (
            entry
            for entry in configured
            if x_api_key is not None
            and hmac.compare_digest(
                x_api_key.encode("utf-8"), entry["key"].encode("utf-8")
            )
        ),
        None,
    )
    if matched is None:
        raise HTTPException(status_code=401, detail="Invalid API credentials.")
    if x_workspace_id is None or WORKSPACE_PATTERN.fullmatch(x_workspace_id) is None:
        raise HTTPException(status_code=422, detail="Invalid workspace ID.")
    role = matched["workspaces"].get(x_workspace_id)
    if role is None:
        raise HTTPException(status_code=403, detail="Workspace access denied.")
    try:
        allowed = state.rate_limit(x_workspace_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Access control is temporarily unavailable."
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Workspace request limit exceeded.",
            headers={"Retry-After": "60"},
        )
    return RequestContext(workspace_id=x_workspace_id, role=role)


RequestContextDep = Annotated[RequestContext, Depends(require_request_context)]
