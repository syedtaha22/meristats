# meristats

meristats runs fixed tests on the Meridian chess engine and publishes the results as a static
website. It builds the engine from a committed SHA, drives it over UCI, and records every run with
its provenance in a local database.

## Status

The repository holds the scaffold: the `meristats` command, the site generator, and its tests. The
published site is at https://syedtaha22.github.io/meristats.

## Installation

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Usage

```bash
.venv/bin/meristats --version
.venv/bin/meristats publish
```

`meristats publish` writes the site into `site-build/`, the working folder of the branch `site`.

## Branches

- `main` holds the code.
- `site` holds the generated site, which GitHub Pages serves.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
