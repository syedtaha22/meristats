# meristats, a testing and results framework for the Meridian chess engine
# Copyright (C) 2026 Syed Taha
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

"""
The authentication of a GitHub App and the calls of the GitHub API.

The installation of the App on a repository is found by asking GitHub. The private key is read by
`openssl`, and nothing is written to disk. The standard library is all that is needed.
"""

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


def b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def app_jwt(app_id, key_path, now=None):
    """
    A signed token that says who the App is, valid for ten minutes. GitHub accepts it only to
    ask for an installation token.
    """
    now = int(time.time() if now is None else now)
    header = b64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = b64url(json.dumps({"iat": now - 60, "exp": now + 540, "iss": str(app_id)},
                                separators=(",", ":")).encode())
    signing = f"{header}.{payload}".encode()
    done = subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(key_path)],
                          input=signing, capture_output=True)

    if done.returncode != 0:
        raise GitHubError("openssl cannot sign with the key: " + done.stderr.decode()[-200:])

    return f"{header}.{payload}.{b64url(done.stdout)}"


class Api:
    """
    Calls of the API with one bearer token. `opener` is the function that opens a request, which
    a test replaces.
    """

    def __init__(self, token, opener=urllib.request.urlopen):
        self.token = token
        self.opener = opener

    def request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.token}",
                   "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "meridian-services"}

        if data is not None:
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(API + path, data=data, method=method, headers=headers)

        try:
            with self.opener(request, timeout=60) as reply:
                text = reply.read().decode()
        except urllib.error.HTTPError as error:
            raise GitHubError(f"{method} {path}: {error.code} {error.read().decode()[:300]}") from error
        except OSError as error:
            raise GitHubError(f"{method} {path}: {error}") from error

        return json.loads(text) if text else {}


def installation_token(app_id, key_path, repository, opener=urllib.request.urlopen):
    """
    A token for the installation of the App on `repository` (OWNER/NAME), valid for one hour.
    The installation is the one that GitHub names for the repository.
    """
    api = Api(app_jwt(app_id, key_path), opener)
    installation = api.request("GET", f"/repos/{repository}/installation")["id"]
    return api.request("POST", f"/app/installations/{installation}/access_tokens")["token"]


def app_identity(app_id, key_path, opener=urllib.request.urlopen):
    """
    The name and id of the App, as GitHub reports them for the signed token.
    """
    reply = Api(app_jwt(app_id, key_path), opener).request("GET", "/app")
    return reply["slug"], reply["id"]
