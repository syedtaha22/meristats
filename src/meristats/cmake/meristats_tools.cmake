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

# Adds the training and tuning tools of meristats to a build of the engine. It is included right
# after the project() call of the engine with -DCMAKE_PROJECT_INCLUDE=<this file>, and the
# deferred call runs when the engine's own targets exist, so that the tools are compiled with the
# flags of the engine and link to its library chess_core.
#
# MERISTATS_TOOLS      the tools to build, separated by semicolons
# MERISTATS_TOOLS_DIR  the folder with the sources of the tools

function(meristats_add_tools)
  foreach(name IN LISTS MERISTATS_TOOLS)
    add_executable(${name} ${MERISTATS_TOOLS_DIR}/${name}.cpp)

    target_link_libraries(${name} PRIVATE chess_core Threads::Threads)

    target_compile_options(${name} PRIVATE ${WARNING_FLAGS})
  endforeach()
endfunction()

cmake_language(DEFER CALL meristats_add_tools)
