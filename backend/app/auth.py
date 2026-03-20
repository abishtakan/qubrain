from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import Header, HTTPException

from .config import AUTH_SESSION_HOURS, CLINICIAN_DISPLAY_NAME, CLINICIAN_PASSWORD, CLINICIAN_USERNAME


@dataclass
class Session:
    username: str
    display_name: str
    expires_at: datetime


class SimpleSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def _cleanup(self) -> None:
        now = datetime.now(UTC)
        expired = [token for token, session in self._sessions.items() if session.expires_at <= now]
        for token in expired:
            self._sessions.pop(token, None)

    def login(self, username: str, password: str) -> dict[str, str | int]:
        if username != CLINICIAN_USERNAME or password != CLINICIAN_PASSWORD:
            raise HTTPException(status_code=401, detail="Invalid username or password.")

        self._cleanup()
        token = secrets.token_urlsafe(32)
        session = Session(
            username=username,
            display_name=CLINICIAN_DISPLAY_NAME,
            expires_at=datetime.now(UTC) + timedelta(hours=AUTH_SESSION_HOURS),
        )
        self._sessions[token] = session
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": session.username,
            "display_name": session.display_name,
            "expires_in_hours": AUTH_SESSION_HOURS,
        }

    def get_user(self, authorization: str | None) -> dict[str, str]:
        self._cleanup()
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required.")

        token = authorization.removeprefix("Bearer ").strip()
        session = self._sessions.get(token)
        if session is None:
            raise HTTPException(status_code=401, detail="Session is invalid or expired.")

        return {"username": session.username, "display_name": session.display_name}

    def logout(self, authorization: str | None) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required.")
        token = authorization.removeprefix("Bearer ").strip()
        self._sessions.pop(token, None)


session_manager = SimpleSessionManager()


def require_current_user(authorization: str | None = Header(default=None)) -> dict[str, str]:
    return session_manager.get_user(authorization)
