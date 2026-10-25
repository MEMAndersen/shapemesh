# Copyright (C) 2026 Mads Kruse
#
# This file is part of shapemesh.
#
# shapemesh is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# shapemesh is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with shapemesh. If not, see <https://www.gnu.org/licenses/>.

import threading

from shapemesh import Cube, MeshSettings, build


def test_build_from_worker_thread_raises_without_gmsh():
    errors = []

    def run():
        try:
            build(Cube((0, 0, 0), 1, 1, 1), MeshSettings())
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert errors and "main thread" in errors[0]


def test_build_from_worker_threads_when_gmsh_initialized():
    import gmsh

    gmsh.initialize()
    try:
        errors = []

        def run():
            try:
                mesh = build(Cube((0, 0, 0), 1, 1, 1), MeshSettings())
                assert mesh.num_elements > 0
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
    finally:
        gmsh.finalize()
