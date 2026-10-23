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
| `lichess_user` | The Lichess account whose games the nightly cycle learns from. Optional. |
| `networks_branch` | The branch of the engine repository that holds the networks. Default `nightly-networks`. |
| `anchor_positions` | A data file of labelled positions that every night's training includes a sample of. Optional. |

## Commands

| Command | Does |
|---|---|
| `build REF` | Builds the engine at a commit and records its hash and `bench` node count. |
| `run puzzles` | Plays the solutions of Lichess puzzles. |
| `run search-nodes` | Searches a suite of 15 positions to fixed depths. |
| `run rules` | Counts the nodes of reference positions (perft) and measures the speed. |
| `run match` | Plays two engines against each other, with a sequential test. |
| `run ladder` | Plays Stockfish at several levels and fits a rating. |
| `run positions` | Plays the engine against itself and writes labelled positions. |
| `run reanalyse` | Searches the games of the Lichess account again, deeper, and writes training data, mistakes and experience. |
| `run train` | Trains a network from labelled positions, and checks that the engine loads it. |
| `run job` | Builds and runs a C++ tool from `tools/` against the engine at a commit. |
| `magics` | Finds the magic numbers of the sliders and writes them as C++ source. |
| `compare BASE RUN...` | Compares puzzle runs with a base run, with a paired test. |
| `report RUN`, `runs`, `pins` | Print a run, list the runs, list the pinned runs. |
| `pin`, `unpin` | Put runs on a page of the site, or take them off. |
| `publish`, `serve` | Write the site, and serve it on this computer. |
| `lichess sync` | Stores the new games of the account in the data folder. |
| `experience info FILE` | Prints the size of an experience file. |
| `network source` | Writes a network file as the C++ source of the engine. |
| `nightly` | Runs the nightly cycle. |
| `fetch stockfish`, `fetch book` | Download Stockfish and the opening book, with their hashes checked. |
| `backup DEST`, `verify` | Copy the data folder, and check every blob against its hash. |

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

## Nightly cycle

`meristats nightly` is one night of learning for the engine. It builds the engine at the head of
`networks_branch`, which is the deployed engine, and then:

1. plays the engine against itself for `--selfplay-minutes`
2. stores the new games of `lichess_user` and searches each position of them again at
   `--analysis-depth`, which gives quiet positions for training, the mistakes (moves that gave
   away at least 100 centipawns) and the best moves for the experience file
3. merges the best moves into `data_dir/experience/experience.bin`, which the engine reads with
   its option `ExperienceFile`, keeping the deeper analysis of a position
4. stores the night in the replay buffer, `data_dir/replay`, and writes the training file from
   the night, a sample of the nights before it, the mistakes of all nights (three times each) and
   a sample of `anchor_positions`
5. trains a network that starts from the deployed one, at a low learning rate
6. plays the new network against the deployed one at `--match-nodes` nodes a move, with a
   sequential test

The night ends with the status `passed` when the test accepts the new network, and
`summary.json` then names `candidate` and `deployed`. The night does not change the engine
repository. A night that reaches its `--deadline` ends with the status `late`. Every stage is a
run in the database, and `data_dir/nightly/<night>/summary.json` holds the numbers of the night.

## Tests

```bash
pytest
```

The tests build a tiny engine repository with CMake, use an oracle engine with scripted answers,
and use a fake fastchess. Tests marked `slow` compare against a real engine and run when
`MERISTATS_ENGINE_REPO` names its repository.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
