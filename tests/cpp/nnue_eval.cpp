/*
  meristats, a testing and results framework for the Meridian chess engine
  Copyright (C) 2026 Syed Taha

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

// Prints the integer evaluation of the engine, in centipawns for the side to move, of each FEN
// that it reads from the standard input, with the network file of the first argument.

#include <iostream>
#include <string>

#include "nnue.hpp"
#include "position.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }

    const auto network = chess::nnue::load(argv[1]);

    if (!network) {
        std::cerr << "the engine cannot load " << argv[1] << "\n";
        return 1;
    }

    std::string line;

    while (std::getline(std::cin, line)) {
        const auto position = chess::Position::from_fen(line);

        if (!position) {
            std::cout << "invalid\n";
            continue;
        }

        chess::nnue::Accumulator accumulator;
        chess::nnue::refresh(*network, *position, accumulator);
        std::cout << chess::nnue::evaluate(*network, accumulator, position->side_to_move())
                  << "\n";
    }

    return 0;
}
