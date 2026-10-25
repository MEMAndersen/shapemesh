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
