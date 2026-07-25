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
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen
import webbrowser

from lumen_engine.models import MediaIdentity

SPOTIFY_ACCOUNTS = "https://accounts.spotify.com"
SPOTIFY_API = "https://api.spotify.com/v1"
SPOTIFY_SCOPES = (
    "user-read-currently-playing "
    "user-read-playback-state "
    "user-modify-playback-state "
    "playlist-read-private "
    "playlist-read-collaborative "
    "user-read-private"
)


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
                    "show_dialog": "true",
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


class SpotifyWebAPI:
    """Small Web API client for Lumen's private Spotify Connect console."""

    def __init__(
        self,
        token_supplier: Callable[[], SpotifyToken],
        timeout_s: float = 10.0,
    ) -> None:
        self.token_supplier = token_supplier
        self.timeout_s = timeout_s
        self.last_playback_payload: dict[str, Any] | None = None

    def console(
        self,
        query: str = "",
        playlist_id: str = "",
    ) -> dict[str, Any]:
        playback = self._request("/me/player")
        self.last_playback_payload = (
            playback if isinstance(playback, dict) else None
        )
        devices_payload = self._request("/me/player/devices") or {}
        profile_payload = self._request("/me") or {}
        token = self.token_supplier()
        granted = set(token.scope.split())
        library_scopes = {
            "playlist-read-private",
            "playlist-read-collaborative",
        }
        library_authorized = library_scopes.issubset(granted)
        result: dict[str, Any] = {
            "connected": True,
            "control_authorized": "user-modify-playback-state" in granted,
            "library_authorized": library_authorized,
            "granted_scopes": sorted(granted),
            "profile": spotify_profile_summary(profile_payload),
            "playback": spotify_playback_summary(playback),
            "devices": [
                spotify_device_summary(device)
                for device in devices_payload.get("devices", [])
                if isinstance(device, dict)
            ],
            "playlists": [],
            "selected_playlist": None,
            "playlist_tracks": [],
            "playlist_error": None,
            "results": [],
            "query": query,
            "playlist_id": playlist_id,
            "observed_at_unix_ms": round(time.time() * 1000),
        }
        if library_authorized:
            playlists_payload = self._request(
                "/me/playlists",
                query={"limit": 50, "offset": 0},
            ) or {}
            result["playlists"] = [
                spotify_playlist_summary(playlist)
                for playlist in playlists_payload.get("items", [])
                if isinstance(playlist, dict)
            ]
        if playlist_id.strip() and library_authorized:
            safe_playlist_id = playlist_id.strip()
            if not safe_playlist_id.isalnum():
                raise ValueError("invalid Spotify playlist ID")
            result["selected_playlist"] = next(
                (
                    playlist
                    for playlist in result["playlists"]
                    if playlist.get("id") == safe_playlist_id
                ),
                None,
            )
            try:
                items_payload = self._request(
                    f"/playlists/{quote(safe_playlist_id)}/items",
                    query={
                        "limit": 50,
                        "offset": 0,
                        "additional_types": "track",
                    },
                ) or {}
                result["playlist_tracks"] = [
                    spotify_track_summary(item)
                    for entry in items_payload.get("items", [])
                    if isinstance(entry, dict)
                    for item in [entry.get("item") or entry.get("track")]
                    if isinstance(item, dict)
                    and item.get("type", "track") == "track"
                ]
            except RuntimeError as error:
                # Spotify's current API limits item retrieval to playlists the
                # account owns or collaborates on. Spotify itself can still
                # open and play the playlist context.
                result["playlist_error"] = str(error)
        if query.strip():
            search = self._request(
                "/search",
                query={"q": query.strip(), "type": "track", "limit": 10},
            ) or {}
            tracks = search.get("tracks", {})
            if isinstance(tracks, dict):
                result["results"] = [
                    spotify_track_summary(track)
                    for track in tracks.get("items", [])
                    if isinstance(track, dict)
                ]
        return result

    def command(self, action: str, payload: dict[str, Any]) -> None:
        device_id = str(payload.get("device_id", "")).strip() or None
        device_query = {"device_id": device_id} if device_id else None
        if action == "play":
            uri = str(payload.get("uri", "")).strip()
            context_uri = str(payload.get("context_uri", "")).strip()
            offset_uri = str(payload.get("offset_uri", "")).strip()
            if context_uri:
                body: dict[str, object] | None = {
                    "context_uri": context_uri
                }
                if offset_uri:
                    body["offset"] = {"uri": offset_uri}
            else:
                body = {"uris": [uri]} if uri else None
            self._request(
                "/me/player/play",
                method="PUT",
                query=device_query,
                body=body,
            )
        elif action == "pause":
            self._request("/me/player/pause", method="PUT", query=device_query)
        elif action == "next":
            self._request("/me/player/next", method="POST", query=device_query)
        elif action == "previous":
            self._request("/me/player/previous", method="POST", query=device_query)
        elif action == "seek":
            position_ms = max(0, int(payload.get("position_ms", 0)))
            query = {"position_ms": position_ms}
            if device_id:
                query["device_id"] = device_id
            self._request("/me/player/seek", method="PUT", query=query)
        elif action == "volume":
            volume = max(0, min(100, int(payload.get("volume_percent", 0))))
            query = {"volume_percent": volume}
            if device_id:
                query["device_id"] = device_id
            self._request("/me/player/volume", method="PUT", query=query)
        elif action == "transfer":
            if not device_id:
                raise ValueError("device_id is required to transfer playback")
            self._request(
                "/me/player",
                method="PUT",
                body={
                    "device_ids": [device_id],
                    "play": bool(payload.get("play", False)),
                },
            )
        elif action == "queue":
            uri = str(payload.get("uri", "")).strip()
            if not uri:
                raise ValueError("uri is required to add an item to the queue")
            query = {"uri": uri}
            if device_id:
                query["device_id"] = device_id
            self._request("/me/player/queue", method="POST", query=query)
        else:
            raise ValueError(f"unknown Spotify command {action!r}")

    def playback(self) -> dict[str, Any] | None:
        payload = self._request("/me/player")
        return payload if isinstance(payload, dict) else None

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: dict[str, object] | None = None,
        body: dict[str, object] | None = None,
    ) -> Any:
        token = self.token_supplier()
        url = f"{SPOTIFY_API}{path}"
        if query:
            encoded = urlencode(
                {
                    key: value
                    for key, value in query.items()
                    if value is not None and value != ""
                }
            )
            if encoded:
                url = f"{url}?{encoded}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {token.access_token}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read()
                if response.status == 204 or not payload.strip():
                    return None
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    # Player-control endpoints occasionally return a short
                    # non-JSON acknowledgement despite a successful 2xx
                    # status. The command was accepted; there is no provider
                    # payload for Lumen to consume.
                    return None
        except HTTPError as error:
            if error.code == 204:
                return None
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "unknown")
                raise RuntimeError(
                    f"Spotify {method} {path} was rate limited; "
                    f"retry after {retry_after}s"
                ) from error
            try:
                detail = json.loads(error.read().decode("utf-8"))
                message = detail.get("error", {}).get("message")
            except Exception:
                message = None
            raise RuntimeError(
                f"Spotify {method} {path} returned {error.code}"
                + (f": {message}" if message else "")
            ) from error


def spotify_track_summary(track: dict[str, Any]) -> dict[str, Any]:
    album = track.get("album")
    album_payload = album if isinstance(album, dict) else {}
    artists = track.get("artists")
    artist_payload = artists if isinstance(artists, list) else []
    images = album_payload.get("images")
    image_payload = images if isinstance(images, list) else []
    image_url = next(
        (
            str(image.get("url"))
            for image in image_payload
            if isinstance(image, dict) and image.get("url")
        ),
        None,
    )
    return {
        "uri": track.get("uri"),
        "id": track.get("id"),
        "name": track.get("name"),
        "artists": [
            str(artist.get("name"))
            for artist in artist_payload
            if isinstance(artist, dict) and artist.get("name")
        ],
        "album": album_payload.get("name"),
        "duration_ms": track.get("duration_ms"),
        "image_url": image_url,
        "explicit": bool(track.get("explicit")),
        "spotify_url": (
            track.get("external_urls", {}).get("spotify")
            if isinstance(track.get("external_urls"), dict)
            else None
        ),
    }


def spotify_profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    images = profile.get("images")
    image_payload = images if isinstance(images, list) else []
    return {
        "id": profile.get("id"),
        "display_name": profile.get("display_name"),
        "product": profile.get("product"),
        "country": profile.get("country"),
        "image_url": next(
            (
                str(image.get("url"))
                for image in image_payload
                if isinstance(image, dict) and image.get("url")
            ),
            None,
        ),
        "spotify_url": (
            profile.get("external_urls", {}).get("spotify")
            if isinstance(profile.get("external_urls"), dict)
            else None
        ),
    }


def spotify_playlist_summary(playlist: dict[str, Any]) -> dict[str, Any]:
    images = playlist.get("images")
    image_payload = images if isinstance(images, list) else []
    owner = playlist.get("owner")
    owner_payload = owner if isinstance(owner, dict) else {}
    item_collection = playlist.get("items")
    if not isinstance(item_collection, dict):
        item_collection = playlist.get("tracks")
    return {
        "id": playlist.get("id"),
        "uri": playlist.get("uri"),
        "name": playlist.get("name"),
        "owner": owner_payload.get("display_name") or owner_payload.get("id"),
        "track_count": (
            item_collection.get("total")
            if isinstance(item_collection, dict)
            else None
        ),
        "image_url": next(
            (
                str(image.get("url"))
                for image in image_payload
                if isinstance(image, dict) and image.get("url")
            ),
            None,
        ),
        "spotify_url": (
            playlist.get("external_urls", {}).get("spotify")
            if isinstance(playlist.get("external_urls"), dict)
            else None
        ),
    }


def spotify_device_summary(device: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": device.get("id"),
        "name": device.get("name"),
        "type": device.get("type"),
        "is_active": bool(device.get("is_active")),
        "is_restricted": bool(device.get("is_restricted")),
        "volume_percent": device.get("volume_percent"),
        "supports_volume": bool(device.get("supports_volume")),
    }


def spotify_playback_summary(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    item = payload.get("item")
    device = payload.get("device")
    return {
        "track": (
            spotify_track_summary(item)
            if isinstance(item, dict)
            else None
        ),
        "device": (
            spotify_device_summary(device)
            if isinstance(device, dict)
            else None
        ),
        "is_playing": bool(payload.get("is_playing")),
        "progress_ms": payload.get("progress_ms"),
        "repeat_state": payload.get("repeat_state"),
        "shuffle_state": bool(payload.get("shuffle_state")),
    }


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
            round(time.time() * 1000)
            if payload.get("progress_ms") is not None
            else None
        ),
        is_playing=bool(payload.get("is_playing")),
        device_name=device_name,
        context_uri=context_uri,
        raw=payload,
    )
