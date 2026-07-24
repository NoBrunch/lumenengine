"""Provider-neutral media identity plus an optional Spotify adapter."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import time
from typing import Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
import webbrowser

from lumen_engine.models import MediaIdentity

SPOTIFY_ACCOUNTS = "https://accounts.spotify.com"
SPOTIFY_API = "https://api.spotify.com/v1"
SPOTIFY_SCOPES = "user-read-currently-playing user-read-playback-state"


class MediaIdentityProvider(Protocol):
    def now_playing(self) -> MediaIdentity | None: ...


@dataclass(slots=True)
class ManualMediaProvider:
    media: MediaIdentity | None = None

    def now_playing(self) -> MediaIdentity | None:
        return self.media


@dataclass(frozen=True, slots=True)
class SpotifyToken:
    access_token: str
    refresh_token: str | None
    expires_at_unix_s: float
    scope: str


class SpotifyTokenCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> SpotifyToken | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return SpotifyToken(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            expires_at_unix_s=float(payload["expires_at_unix_s"]),
            scope=payload.get("scope", ""),
        )

    def save(self, token: SpotifyToken) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "access_token": token.access_token,
                    "refresh_token": token.refresh_token,
                    "expires_at_unix_s": token.expires_at_unix_s,
                    "scope": token.scope,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)


class SpotifyOAuthPKCE:
    """One-user local OAuth flow with no client secret stored in the project."""

    def __init__(
        self,
        client_id: str,
        cache: SpotifyTokenCache,
        redirect_uri: str = "http://127.0.0.1:8765/callback",
    ) -> None:
        self.client_id = client_id
        self.cache = cache
        self.redirect_uri = redirect_uri

    def login(self, open_browser: bool = True, timeout_s: float = 180.0) -> SpotifyToken:
        verifier = secrets.token_urlsafe(72)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        expected_state = secrets.token_urlsafe(24)
        result: dict[str, str] = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(handler_self) -> None:  # noqa: N802
                query = parse_qs(urlparse(handler_self.path).query)
                if "code" in query:
                    result["code"] = query["code"][0]
                if "state" in query:
                    result["state"] = query["state"][0]
                if "error" in query:
                    result["error"] = query["error"][0]
                body = (
                    b"Lumen Engine received the Spotify response. "
                    b"You may close this tab."
                )
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "text/plain; charset=utf-8")
                handler_self.send_header("Content-Length", str(len(body)))
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(handler_self, *_: object) -> None:
                return

        parsed_redirect = urlparse(self.redirect_uri)
        server = HTTPServer(
            (parsed_redirect.hostname or "127.0.0.1", parsed_redirect.port or 8765),
            CallbackHandler,
        )
        server.timeout = timeout_s
        authorization_url = (
            f"{SPOTIFY_ACCOUNTS}/authorize?"
            + urlencode(
                {
                    "client_id": self.client_id,
                    "response_type": "code",
                    "redirect_uri": self.redirect_uri,
                    "scope": SPOTIFY_SCOPES,
                    "code_challenge_method": "S256",
                    "code_challenge": challenge,
                    "state": expected_state,
                }
            )
        )
        if open_browser:
            webbrowser.open(authorization_url)
        else:
            print(authorization_url)
        try:
            server.handle_request()
        finally:
            server.server_close()

        if result.get("error"):
            raise RuntimeError(f"Spotify authorization failed: {result['error']}")
        if result.get("state") != expected_state or "code" not in result:
            raise RuntimeError("Spotify authorization timed out or returned invalid state")

        token = self._exchange(
            {
                "client_id": self.client_id,
                "grant_type": "authorization_code",
                "code": result["code"],
                "redirect_uri": self.redirect_uri,
                "code_verifier": verifier,
            },
            previous_refresh_token=None,
        )
        self.cache.save(token)
        return token

    def valid_token(self) -> SpotifyToken:
        token = self.cache.load()
        if token is None:
            raise RuntimeError("Spotify is not connected; run `lumen spotify-login`")
        if token.expires_at_unix_s - time.time() > 60:
            return token
        if not token.refresh_token:
            raise RuntimeError("Spotify token expired and has no refresh token")
        refreshed = self._exchange(
            {
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": token.refresh_token,
            },
            previous_refresh_token=token.refresh_token,
        )
        self.cache.save(refreshed)
        return refreshed

    @staticmethod
    def _exchange(
        form: dict[str, str], previous_refresh_token: str | None
    ) -> SpotifyToken:
        request = Request(
            f"{SPOTIFY_ACCOUNTS}/api/token",
            data=urlencode(form).encode("ascii"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        return SpotifyToken(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token", previous_refresh_token),
            expires_at_unix_s=time.time() + int(payload.get("expires_in", 3600)),
            scope=payload.get("scope", SPOTIFY_SCOPES),
        )


class SpotifyNowPlayingProvider:
    def __init__(
        self,
        token_supplier: Callable[[], SpotifyToken],
        timeout_s: float = 10.0,
    ) -> None:
        self.token_supplier = token_supplier
        self.timeout_s = timeout_s

    def now_playing(self) -> MediaIdentity | None:
        token = self.token_supplier()
        request = Request(
            f"{SPOTIFY_API}/me/player",
            headers={"Authorization": f"Bearer {token.access_token}"},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                if response.status == 204:
                    return None
                payload = json.load(response)
        except HTTPError as error:
            if error.code == 204:
                return None
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "unknown")
                raise RuntimeError(
                    f"Spotify rate limited this request; retry after {retry_after}s"
                ) from error
            raise
        return media_identity_from_spotify(payload)


def media_identity_from_spotify(payload: dict[str, object]) -> MediaIdentity | None:
    item = payload.get("item")
    if not isinstance(item, dict):
        return None
    artists_payload = item.get("artists", [])
    artists = tuple(
        str(artist.get("name"))
        for artist in artists_payload
        if isinstance(artist, dict) and artist.get("name")
    )
    album_payload = item.get("album")
    album = (
        str(album_payload.get("name"))
        if isinstance(album_payload, dict) and album_payload.get("name")
        else None
    )
    device_payload = payload.get("device")
    device_name = (
        str(device_payload.get("name"))
        if isinstance(device_payload, dict) and device_payload.get("name")
        else None
    )
    context_payload = payload.get("context")
    context_uri = (
        str(context_payload.get("uri"))
        if isinstance(context_payload, dict) and context_payload.get("uri")
        else None
    )
    identifier = item.get("uri") or item.get("id")
    return MediaIdentity(
        provider="spotify",
        provider_item_id=str(identifier) if identifier else None,
        title=str(item.get("name")) if item.get("name") else None,
        artists=artists,
        album=album,
        duration_ms=int(item["duration_ms"]) if item.get("duration_ms") else None,
        observed_position_ms=(
            int(payload["progress_ms"]) if payload.get("progress_ms") is not None else None
        ),
        observed_at_unix_ms=(
            int(payload["timestamp"]) if payload.get("timestamp") is not None else None
        ),
        is_playing=bool(payload.get("is_playing")),
        device_name=device_name,
        context_uri=context_uri,
        raw=payload,
    )

