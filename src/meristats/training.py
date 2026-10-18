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

"""
Training of the network of the engine from labelled positions, with PyTorch.

The data file has lines `FEN<TAB>result<TAB>game<TAB>score`, where the result is the points of
White and the score is in centipawns for the side to move. The target of a position is `lambda`
times the expected points of the search score plus `1 - lambda` times the result of the game,
from the side to move. A fifth of the games, those whose number divides by 5, are held out.
"""

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from meristats import nnue

CURVE = 1.25
CHUNK_BYTES = 8 * 1024 * 1024
HELD_FENS = 4000


class TrainingError(RuntimeError):
    pass


@dataclass
class Dataset:
    """
    The positions of a data file as arrays. A row holds up to 32 pieces as `piece * 64 + square`
    with -1 after the last, the side to move, the king bucket and the mirror of each point of
    view, the number of pieces, and the target.
    """

    pieces: np.ndarray  # int16, N x 32
    context: np.ndarray  # uint8, N x 5: side, bucket white, bucket black, mirror white, black
    count: np.ndarray  # uint8, N
    target: np.ndarray  # float32, N
    held: np.ndarray  # bool, N
    held_fens: list[str] = field(default_factory=list)
    read: int = 0  # the lines that were read
    skipped: int = 0  # the lines that were left out

    def __len__(self) -> int:
        return len(self.target)


def target_of(score: int, result: float, side: int, lam: float) -> float:
    """
    The target of a position: the expected points of the score and the result, from the side to
    move.
    """
    side_cp = score if side == 0 else -score
    side_result = result if side == 0 else 1.0 - result
    points = 1.0 / (1.0 + 10.0 ** (-CURVE * side_cp / 400.0))
    return lam * points + (1.0 - lam) * side_result


def _convert(args: tuple[Path, int, int, float, int]) -> tuple:
    path, start, end, lam, max_score = args
    pieces, context, count, target, held, fens = [], [], [], [], [], []
    skipped = 0
    lines = 0

    with open(path, "rb") as handle:
        if start:
            handle.seek(start - 1)
            handle.readline()  # the rest of the line that the previous chunk ends with

        while handle.tell() <= end:
            raw = handle.readline()

            if not raw:
                break

            lines += 1
            fields = raw.decode().rstrip("\n").split("\t")

            if len(fields) < 4 or not fields[3]:
                skipped += 1
                continue

            fen, result, game, score = fields[0], float(fields[1]), int(fields[2]), int(fields[3])

            if abs(score) > max_score:
                skipped += 1
                continue

            try:
                placed, side = nnue.parse_board(fen)
            except (ValueError, IndexError):
                skipped += 1
                continue

            kinds = [piece for piece, _ in placed]

            # Each side needs one king, and a position has at most 32 pieces.
            if kinds.count(5) != 1 or kinds.count(11) != 1 or len(placed) > nnue.MAX_PIECES:
                skipped += 1
                continue

            # A position in check has no quiet score, and the search does not evaluate it.
            if nnue.side_in_check(placed, side):
                skipped += 1
                continue

            placed.sort(key=lambda item: item[1])

            row = [-1] * nnue.MAX_PIECES

            for index, (piece, square) in enumerate(placed):
                row[index] = piece * 64 + square

            kings = {piece: square for piece, square in placed if piece in (5, 11)}
            bucket_white, mirror_white = nnue.king_context(0, kings[5])
            bucket_black, mirror_black = nnue.king_context(1, kings[11])
            pieces.append(row)
            context.append((side, bucket_white, bucket_black, mirror_white, mirror_black))
            count.append(len(placed))
            target.append(target_of(score, result, side, lam))
            is_held = game % 5 == 0
            held.append(is_held)

            if is_held and len(fens) < HELD_FENS:
                fens.append(fen)

    return (
        np.array(pieces, np.int16).reshape(-1, nnue.MAX_PIECES),
        np.array(context, np.uint8).reshape(-1, 5),
        np.array(count, np.uint8),
        np.array(target, np.float32),
        np.array(held, bool),
        fens,
        lines,
        skipped,
    )


def read_dataset(
    path: Path,
    *,
    lam: float = 0.7,
    max_score: int = 2500,
    limit: int = 0,
    workers: int | None = None,
) -> Dataset:
    """
    Reads a data file with a pool of processes. `limit` keeps the first positions of the file, and
    0 keeps all of them.
    """
    path = Path(path)
    size = path.stat().st_size
    workers = workers or os.cpu_count() or 1
    ranges = [
        (path, start, min(start + CHUNK_BYTES, size) - 1, lam, max_score)
        for start in range(0, size, CHUNK_BYTES)
    ]
    parts = []
    kept = 0
    lines = skipped = 0

    with Pool(workers) as pool:
        for part in pool.imap(_convert, ranges):
            parts.append(part)
            kept += len(part[3])
            lines += part[6]
            skipped += part[7]

            if limit and kept >= limit:
                pool.terminate()
                break

    if not parts:
        raise TrainingError(f"{path} has no positions")

    pieces = np.concatenate([p[0] for p in parts])
    context = np.concatenate([p[1] for p in parts])
    count = np.concatenate([p[2] for p in parts])
    target = np.concatenate([p[3] for p in parts])
    held = np.concatenate([p[4] for p in parts])
    fens = [fen for p in parts for fen in p[5]][:HELD_FENS]

    if limit:
        pieces, context, count, target, held = (
            pieces[:limit],
            context[:limit],
            count[:limit],
            target[:limit],
            held[:limit],
        )

    return Dataset(pieces, context, count, target, held, fens, lines, skipped)


@dataclass
class Result:
    network: nnue.Network
    best_held: float
    epochs: list[dict]
    seconds_loading: float
    seconds_training: float
    device: str
    quantized_difference: float | None = None
    quantized_magnitude: float | None = None
    checked: int = 0


def train(
    data: Dataset,
    *,
    epochs: int = 30,
    batch: int = 16384,
    lr: float = 0.001,
    seed: int = 1,
    hidden: int = nnue.HIDDEN,
    init: nnue.Network | None = None,
    device: str | None = None,
    log: Callable[[str], None] = print,
    seconds_loading: float = 0.0,
) -> Result:
    """
    Trains a network with Adam on the positions that are not held out, and keeps the weights of the
    epoch with the lowest held-out error. The learning rate falls from `lr` to a tenth of it over
    the epochs. Returns the best network, quantized.
    """
    import torch
    import torch.nn.functional as functional

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    train_rows = np.flatnonzero(~data.held)
    held_rows = np.flatnonzero(data.held)

    if len(train_rows) < batch or not len(held_rows):
        raise TrainingError("there are too few positions to train and to hold out")

    pieces = torch.from_numpy(data.pieces).to(device)
    context = torch.from_numpy(data.context).to(device)
    count = torch.from_numpy(data.count).to(device)
    target = torch.from_numpy(data.target).to(device)
    train_index = torch.from_numpy(train_rows).to(device)
    held_index = torch.from_numpy(held_rows).to(device)
    pad = nnue.INPUTS

    if init is not None:
        w0, b0, wo0, bo0 = (torch.from_numpy(a) for a in nnue.to_float(init))
        hidden = init.hidden
        w = torch.cat([w0, torch.zeros(1, hidden)]).to(device)
        b, wo, bo = b0.to(device), wo0.to(device), bo0.to(device)
    else:
        w = torch.randn(nnue.INPUTS + 1, hidden, device=device, generator=generator) * 0.1
        w[pad] = 0
        b = torch.zeros(hidden, device=device)
        wo = torch.randn(nnue.BUCKETS, 2, hidden, device=device, generator=generator) * 0.1
        bo = torch.zeros(nnue.BUCKETS, device=device)

    parameters = [p.requires_grad_() for p in (w, b, wo, bo)]
    optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    to_logit = nnue.OUTPUT_SCALE * CURVE * math.log(10.0) / 400.0

    def forward(rows: torch.Tensor) -> torch.Tensor:
        placed = pieces[rows].long()
        valid = placed >= 0
        placed = placed.clamp(min=0)
        piece, square = placed // 64, placed % 64
        ctx = context[rows].long()
        side = ctx[:, 0]
        white = ctx[:, 1:2] * nnue.BASE_INPUTS + piece * 64 + (square ^ (ctx[:, 3:4] * 7))
        black = (
            ctx[:, 2:3] * nnue.BASE_INPUTS
            + ((piece + 6) % 12) * 64
            + ((square ^ 56) ^ (ctx[:, 4:5] * 7))
        )
        white = torch.where(valid, white, pad)
        black = torch.where(valid, black, pad)
        acc_white = functional.embedding_bag(white, w, mode="sum", padding_idx=pad) + b
        acc_black = functional.embedding_bag(black, w, mode="sum", padding_idx=pad) + b
        first = (side == 0).unsqueeze(1)
        us = torch.where(first, acc_white, acc_black).clamp(0, 1)
        them = torch.where(first, acc_black, acc_white).clamp(0, 1)
        bucket = ((count[rows].long() - 2) // (32 // nnue.BUCKETS)).clamp(0, nnue.BUCKETS - 1)
        weights = wo[bucket]
        return (us * weights[:, 0]).sum(1) + (them * weights[:, 1]).sum(1) + bo[bucket]

    def loss_of(rows: torch.Tensor) -> torch.Tensor:
        prediction = torch.sigmoid(forward(rows) * to_logit)
        return ((prediction - target[rows]) ** 2).sum()

    def held_loss() -> float:
        total = 0.0

        with torch.no_grad():
            for start in range(0, len(held_index), 1 << 16):
                total += float(loss_of(held_index[start : start + (1 << 16)]))

        return total / len(held_index)

    log(
        f"{len(train_rows)} training and {len(held_rows)} held-out positions on {device}, "
        f"loaded in {seconds_loading:.1f}s"
    )
    best, best_state = 1e9, None
    history = []
    log(f"start: held-out loss {held_loss():.5f}")
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        stamp = time.monotonic()
        rate = lr * 0.1 ** ((epoch - 1) / epochs)

        for group in optimizer.param_groups:
            group["lr"] = rate

        order = train_index[torch.randperm(len(train_index), device=device, generator=generator)]
        running = torch.zeros((), device=device)
        seen = 0

        for start in range(0, len(order) - batch + 1, batch):
            rows = order[start : start + batch]
            loss = loss_of(rows)
            optimizer.zero_grad(set_to_none=True)
            (loss / batch).backward()
            optimizer.step()
            running += loss.detach()
            seen += batch

        held = held_loss()
        train_loss = float(running) / max(1, seen)
        seconds = time.monotonic() - stamp
        history.append(
            {"epoch": epoch, "lr": rate, "train": train_loss, "held": held, "s": seconds}
        )
        log(
            f"epoch {epoch:2d}: lr {rate:.5f} train {train_loss:.5f} held {held:.5f} "
            f"({seconds:.1f}s)"
        )

        if held < best:
            best = held
            best_state = [p.detach().cpu().numpy().copy() for p in (w, b, wo, bo)]

    training_seconds = time.monotonic() - started
    network = nnue.quantize(
        best_state[0][: nnue.INPUTS], best_state[1], best_state[2], best_state[3]
    )
    result = Result(network, best, history, seconds_loading, training_seconds, device)
    _compare(result, best_state, data.held_fens)
    log(
        f"quantized against float: mean difference {result.quantized_difference:.2f}cp over "
        f"{result.checked} positions (mean magnitude {result.quantized_magnitude:.1f}cp)"
    )
    log(f"best held-out loss {best:.5f}")
    return result


def _compare(result: Result, state: list[np.ndarray], fens: list[str]) -> None:
    """
    The integer network against the floating point one on held-out positions.
    """
    w, b, wo, bo = state
    difference = magnitude = 0.0

    for fen in fens:
        pieces, side = nnue.parse_board(fen)
        kings = {piece: square for piece, square in pieces if piece in (5, 11)}
        accumulators = []

        for perspective, king in ((0, kings[5]), (1, kings[11])):
            bucket, mirror = nnue.king_context(perspective, king)
            total = b.astype(np.float64).copy()

            for piece, square in pieces:
                total += w[nnue.king_feature(perspective, piece, square, bucket, mirror)]

            accumulators.append(np.clip(total, 0, 1))

        layer = nnue.bucket_of(len(pieces))
        output = (
            bo[layer]
            + accumulators[side] @ wo[layer, 0].astype(np.float64)
            + accumulators[1 - side] @ wo[layer, 1].astype(np.float64)
        )
        reference = float(output) * nnue.OUTPUT_SCALE
        difference += abs(nnue.evaluate(result.network, fen) - reference)
        magnitude += abs(reference)

    result.checked = len(fens)
    result.quantized_difference = difference / max(1, len(fens))
    result.quantized_magnitude = magnitude / max(1, len(fens))
