"""The solver process: it builds the run from the spec, marches it, stores it and streams it to the interface.

The interface starts ``run`` in a separate process (multiprocessing, 'spawn'): the compiled kernels hold the Python lock,
so a thread would freeze the window during a step. Messages on ``messages`` (all plain data):

    ("started", run_dir, info)              info: count, labels
    ("step", index, record, seconds)        every step
    ("failed", index, module, error)        the run ends there; the steps before it are kept
    ("error", text)                         the set-up was refused (nothing ran)
    ("finished", state)                     complete / stopped

The newest 3D frame goes to ``frames`` (a queue of size one: the worker never waits for the window; a frame the window
has not taken yet is replaced by the newer one).
"""
import queue
import traceback


def _offer(frames, frame):
    try:
        frames.put_nowait(frame)
    except queue.Full:
        try:
            frames.get_nowait()
        except queue.Empty:
            pass
        try:
            frames.put_nowait(frame)
        except queue.Full:
            pass


def run(spec_dict, run_dir, messages, frames, pause, stop, threads=0):
    try:
        from . import resources
        resources.ensure_library()
        if threads:
            import numba
            numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
        from . import engine
        from .runstore import RunWriter
        from .spec import Spec
        spec = Spec.from_dict(spec_dict)
        built = engine.build(spec)
        writer = RunWriter(run_dir, spec, built)
    except Exception as error:  # noqa: BLE001
        messages.put(("error", f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"))
        return
    messages.put(("started", str(run_dir), dict(count=built.count, method=spec.method, compute=spec.compute)))
    state = "complete"
    try:
        for item in engine.march(built, should_stop=stop.is_set):
            if isinstance(item, engine.Failure):
                writer.close("failed", f"step {item.index} ({item.module}): {item.error}")
                messages.put(("failed", item.index, item.module, item.error))
                return
            writer.append(item.record, item.seconds)
            last = item.index == built.count
            frame = engine.frame(built, item)
            if item.index % int(spec.frame_every) == 0 or last:
                writer.save_frame(frame)
            _offer(frames, frame)
            messages.put(("step", item.index, item.record, item.seconds))
            while pause.is_set() and not stop.is_set():
                stop.wait(0.1)
        if stop.is_set() and writer.steps < built.count+1:
            state = "stopped"
    except Exception as error:  # noqa: BLE001  (a defect, not a failed step: report it with its trace)
        writer.close("failed", f"{type(error).__name__}: {error}")
        messages.put(("error", f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=5)}"))
        return
    writer.close(state)
    messages.put(("finished", state))
