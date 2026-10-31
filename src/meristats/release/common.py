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
What the steps that touch a repository on GitHub share: their error, the commands of git, and the
environment that gives git the token of the App for one command.
"""

import base64
import os
import subprocess


class ReleaseError(RuntimeError):
    pass


def git(folder, *arguments, env=None, check=True):
    done = subprocess.run(
        ["git", "-C", str(folder), *arguments], capture_output=True, text=True, env=env
    )

    if check and done.returncode != 0:
        raise ReleaseError(
            f"git {' '.join(arguments[:3])} failed: {(done.stderr or done.stdout)[-300:]}"
        )

    return done.stdout.strip()


def auth_environment(token):
    """
    The environment that gives git the token for one command. It is not written to the
    configuration of any repository, and it does not show in the arguments of the process.
    """
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: basic {encoded}",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment
