from meristats.cli import main
from meristats.publish import MESSAGE, TITLE, publish


def test_publish_writes_index_and_marker(tmp_path):
    written = publish(tmp_path)

    assert {path.name for path in written} == {"index.html", ".nojekyll"}
    assert (tmp_path / ".nojekyll").read_text() == ""

    page = (tmp_path / "index.html").read_text()
    assert f"<title>{TITLE}</title>" in page
    assert MESSAGE in page


def test_publish_overwrites_an_existing_site(tmp_path):
    (tmp_path / "index.html").write_text("old")
    publish(tmp_path)

    assert (tmp_path / "index.html").read_text() != "old"


def test_command_publish_writes_to_out(tmp_path, capsys):
    out = tmp_path / "site"

    assert main(["publish", "--out", str(out)]) == 0
    assert (out / "index.html").exists()
    assert str(out / "index.html") in capsys.readouterr().out
