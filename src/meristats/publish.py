from pathlib import Path

TITLE = "meristats"
MESSAGE = "No results are published yet."

INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
</head>
<body>
  <h1>{title}</h1>
  <p>{message}</p>
</body>
</html>
"""


def publish(out_dir: Path) -> list[Path]:
    """
    Writes the site into `out_dir` and returns the paths of the files that it wrote.

    The site is a top-level `index.html` and an empty `.nojekyll`, which stops GitHub Pages from
    running Jekyll on the folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(INDEX.format(title=TITLE, message=MESSAGE), encoding="utf-8")
    marker = out_dir / ".nojekyll"
    marker.write_text("", encoding="utf-8")
    return [index, marker]
