# Meristats

meristats is a testing and results framework for the
[Meridian](https://github.com/syedtaha22/meridian) chess engine. It measures Meridian from version
1.0.0 onwards. It builds the engine from a committed SHA, drives it over UCI, records every run
with its provenance in a local database, and publishes the results as a static website at
https://syedtaha22.github.io/meristats.

## Installation

Python 3.12 or newer, git, cmake and a C++ compiler are required. The commands below run in the
activated environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
cp meristats.toml.example meristats.toml     # then edit the paths
meristats doctor
```

Training a network needs PyTorch: `pip install -e ".[train]"`. The default build for Linux
includes CUDA and is about 3 GB. The tests of the trainer are skipped without it.

`meristats.toml` is read from `--config`, from `$MERISTATS_CONFIG`, or from the current folder. It
is not committed, because it holds the paths of one machine.

| Key | Meaning |
|---|---|
| `engine_repo` | The git repository of the engine. Builds use `git archive`, never the working tree. |
| `data_dir` | The database and the blobs. Not a git repository, and never pushed. |
| `cache_dir` | Sources, builds, downloads and match folders. It can be deleted without losing a result. |
| `workers` | The number of worker processes. It has no default. |
| `fastchess` | The path of fastchess, for matches. Optional. |
| `stockfish` | The path of Stockfish, for the ladder. Optional, and `meristats fetch` can download it. |
| `trainer_python` | The Python that has PyTorch, when it is not the Python of meristats. Optional. |
| `networks_branch` | The branch of the engine repository that holds the networks. Default `nightly-networks`. |
| `anchor_positions` | A data file of labelled positions that every training includes a sample of. Optional. |
| `leela_positions` | A data file from `leela convert`, of which every training includes a sample. Optional. |
| `engine_repository` | `OWNER/NAME` of the engine repository on GitHub, for the pull request. Optional. |
| `site_repository` | `OWNER/NAME` of the repository of the site. Optional. |
| `site_worktree` | The worktree of the branch `site`, which `meristats train` publishes into. Optional. |
| `app_id`, `app_key` | The id of the GitHub App that pushes, and the path of its private key. Optional. |
| `commit_name`, `commit_email` | The author of the commits of the App. Optional. |
| `mail_settings` | A file of lines `SMTP_USER=`, `SMTP_PASSWORD=` and `MAIL_TO=`, for the mail after a cycle. Optional. |

## Commands

| Command | Does |
|---|---|
| `build REF` | Builds the engine at a commit and records its hash and `bench` node count. `--hidden N` builds it for a network of N values for each side. |
| `run puzzles` | Plays the solutions of Lichess puzzles. |
| `run search-nodes` | Searches a suite of 15 positions to fixed depths. |
| `run rules` | Counts the nodes of reference positions (perft) and measures the speed. |
| `run match` | Plays two engines against each other, with a sequential test. |
| `run ladder` | Plays Stockfish at several levels and fits a rating. |
| `run positions` | Plays the engine against itself and writes labelled positions. |
| `run train` | Trains a network from labelled positions, and checks that the engine loads it. `--hidden N` trains a network of another width, with an engine built for it. |
| `run networks SPEC` | Trains candidate networks of several shapes, measures their speed and plays them against the deployed engine. |
| `run job` | Builds and runs a C++ tool from `tools/` against the engine at a commit. |
| `magics` | Finds the magic numbers of the sliders and writes them as C++ source. |
| `compare BASE RUN...` | Compares puzzle runs with a base run, with a paired test. |
| `report RUN`, `runs`, `pins` | Print a run, list the runs, list the pinned runs. |
| `pin`, `unpin` | Put runs on a page of the site, or take them off. |
| `publish`, `serve` | Write the site, and serve it on this computer. |
| `leela fetch RUN` | Downloads an even sample of the tars of Leela Chess Zero from a folder of storage.lczero.org. |
| `leela convert OUT TAR...` | Turns the tars into training positions. |
| `network source` | Writes a network file as the C++ source of the engine. |
| `train [DURATION]` | Trains for a time (`8h`, `90m`, `2d`) or until Ctrl-C, as cycles of data, training and eval, and then publishes. |
| `release check`, `sync`, `promote SUMMARY`, `site` | One step of the release: check the App, sync the branch of the networks with `main`, commit a network, publish the site. |
| `fetch stockfish`, `fetch book` | Download Stockfish and the opening book, with their hashes checked. |
| `backup DEST`, `verify` | Copy the data folder, and check every blob against its hash. |
| `merge DIR` | Add the runs, blobs and pins of another data folder that this one lacks. |

A run id can be shortened to any prefix that names one run. The method of each protocol, with its
version, is on the page Protocols of the site.

Engines are driven only through UCI. The option `Config` selects a search preset with changes, such
as `network+razoring=1`, and an engine without it runs with its own default.

## Checks before a run

A run stops when one of these fails:

- The working tree of meristats has changes, unless `--allow-dirty` gives the run the grade
  `unverified`.
- The engine does not come from a commit, or `bench` no longer gives the recorded node count.
- The engine does not accept an option that it is sent.
- The dataset does not have the recorded hash.
- A search limited by nodes or depth does not run with one thread, or 1,000 puzzles played twice in
  different processes give different outcomes.
- A search with a time limit does not get `MoveOverhead 0`.
- `fastchess --compliance` fails for an engine of a match.

The power source, the governor, the CPU frequency, the load and the speed of the machine are
recorded for every run and are not enforced.

## Data

```
data_dir/
  meristats.db           runs, engines, machines, datasets, blobs, pins
  blobs/ab/<sha256>.zst  files, compressed, named by the hash of their bytes
```

Triggers keep the rows of the database from being changed or removed. A run is `verified` when it
was made from a clean state of meristats and of the engine commit, and `unverified` otherwise.

## Site

`meristats publish` writes the pinned runs into `site-build/`, the worktree of the branch `site`,
which GitHub Pages serves. The pages are `index.html`, one `<page>.html` for each group of pinned
runs, `protocols.html`, `assets/` and `data/results.json`. A page is addressed without `.html`, so
`meristats serve` serves the folder the same way, at http://127.0.0.1:8000.

```bash
meristats pin RUN --page steps --label "Step 10"
meristats compare BASE VARIANT_A VARIANT_B --pin techniques
meristats publish
```

## Training

```bash
meristats run positions --engine REF --games 20000 --depth 7 --out positions.txt
meristats run train --engine REF --data positions.txt --out net.nnue
meristats run puzzles --engine REF --nodes 20000 --eval-file net.nnue
```

The last command, and `run match` with `--a-eval-file`, measure a network before it is installed
into the engine by hand. `train_policy` and `build_book` are C++ tools, because they use the
evaluation, the move features and the position hash of the engine.

## Training cycles

`meristats train 8h` trains for 8 hours, and `meristats train` trains until Ctrl-C. It runs cycles
one after the other. A cycle has three steps:

1. **Data.** The engine at the head of `networks_branch`, which is the deployed engine, plays itself
   at `--selfplay-depth`, in chunks of `--chunk-minutes`. Every chunk is stored in the replay
   buffer, `data_dir/replay`, when it is finished, with its own seed. The step ends when the
   positions that no test has seen reach `--test-after`, or at the end of the duration.
2. **Train.** A network starts from the deployed one and trains on all the new positions, a sample
   of `--history-lines` older ones, a sample of `anchor_positions` and, when set, a sample of
   `leela_positions`. It stops when the held-out loss has not improved for `--patience` epochs, or
   after `--epochs`.
3. **Eval.** The new network plays the deployed one at `--match-nodes` nodes a move, in a
   sequential test with Elo bounds `--elo0` and `--elo1`, for as long as the test needs.

The cycle ends with the status `passed` when the test accepts the network, and `rejected` when it
does not. After a rejection the next cycle waits for `--growth` times as many new positions. The
numbers that outlive a cycle are in `data_dir/train-state.json`, and `data_dir/cycles/<id>/` has
the summary and the files of each cycle. Ctrl-C ends the step that is running at once, and the
chunks that were finished stay in the buffer.

With the release settings in `meristats.toml`, the branch of the networks is brought up to date with
`main` before a cycle (`release sync`). A network that passed is committed by the App and put in
the pull request of the engine repository (`release promote`), and a mail reports every cycle. At
the end, the match of every cycle is pinned on the page Matches, and the site is committed and
pushed by the App (`release site`). Two cycles in a row that fail stop the training.

## Networks of other shapes

The width of the first layer is fixed when the engine is built (`CHESS_NNUE_HIDDEN`, 128 by
default), and a network file holds its width. `meristats build REF --hidden 256` builds an engine
for a wider network, as another build of the same commit, and `run train --hidden 256` and `run
match --a-hidden 256` build it by themselves. An engine of another width has no built-in network
and loads one with `EvalFile`.

`run networks SPEC` runs a list of candidates, for example widths against the amount of data:

```toml
engine = "HEAD"
data = "/data/train.txt"
rounds = 1000            # pairs of games
nodes = 50000            # or: st = 0.05
[[candidate]]
name = "w256"
hidden = 256
epochs = 6
lr = 0.0005
```

For each candidate it builds the engine, trains, measures the speed (nodes a second with the
network) and plays the deployed engine. A match at a fixed number of nodes does not see that a wide
network is slower, so the table has the speed as a share of the speed of the deployed engine, and
`st` plays at a fixed time. The results are in `results.json` next to the spec, and a candidate
that is in it is not run again.

## Leela Chess Zero

`meristats leela fetch k2hybrid-bt4tf13g-v7op1 --count 60` downloads 60 tars (about 260 MB each),
spread evenly over the run so that the sample has games of all its stages, and `meristats leela
convert leela.txt TAR...` turns them into positions in the format of the `positions` protocol.
Setting `leela_positions` to that file adds a sample of it to the training data of every cycle.

## Tests

```bash
pytest
```

The tests build a tiny engine repository with CMake, use an oracle engine with scripted answers,
and use a fake fastchess. Tests marked `slow` compare against a real engine and run when
`MERISTATS_ENGINE_REPO` names its repository.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
