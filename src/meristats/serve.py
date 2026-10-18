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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class SiteHandler(SimpleHTTPRequestHandler):
    """
    Serves a folder as GitHub Pages does: the address `/steps` gives `steps.html` when no file or
    folder of that name exists.
    """

    def translate_path(self, path: str) -> str:
        translated = Path(super().translate_path(path))

        if not translated.exists() and translated.suffix == "":
            page = translated.with_name(translated.name + ".html")

            if page.is_file():
                return str(page)

        return str(translated)


def make_server(directory: Path, port: int = 8000, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """
    Returns a server for the site in `directory`. Port 0 picks a free port, which the server
    reports as `server.server_address`.
    """
    handler = functools.partial(SiteHandler, directory=str(directory))
    return ThreadingHTTPServer((host, port), handler)
