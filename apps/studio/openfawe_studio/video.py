"""OF-0010 revision 19 addendum 7 (A3): a stored run as a video - every stored frame drawn off screen with the window's
scene (``scene.Scene``) and written as an animated GIF with Pillow (on every installation of the studio), or as MP4 through
an ``ffmpeg`` executable on the PATH (revision 19 addendum 15, VID-01: frames piped as raw RGB, H.264 with the yuv420p pixel
format that players expect; ffmpeg is not a dependency of the studio) or, without one, through ``imageio`` with its ffmpeg
plug-in (not a dependency either; its licence notice goes with it). Frames are taken in the order of the stored frames;
one stored frame is one video frame (a GIF of more than ``MAX_GIF_FRAMES`` stored frames takes every n-th, review STR-06).
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np

FFMPEG_CRF = 18               # VID-01: visually lossless H.264 for the papers' supplementary videos


def ffmpeg_executable():
    """The ffmpeg on the PATH, or None."""
    return shutil.which("ffmpeg")


def write_mp4_ffmpeg(images, out, size, fps, executable):
    """Pipe RGB frames to ffmpeg (``-f rawvideo``), H.264, yuv420p; raises RuntimeError with ffmpeg's last lines on failure.
    The frame size must be even in both directions (yuv420p)."""
    width, height = (int(v) for v in size)
    if width % 2 or height % 2:
        raise ValueError("the MP4 frame size must be even in both directions")
    command = [executable, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
               "-crf", str(FFMPEG_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    count = 0
    try:
        for image in images:
            image = np.ascontiguousarray(image, dtype=np.uint8)
            if image.shape != (height, width, 3):
                raise ValueError(f"a frame of {image.shape[1]}x{image.shape[0]} in a {width}x{height} video")
            process.stdin.write(image.tobytes())
            count += 1
    finally:
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", "replace")
        process.stderr.close()
        code = process.wait()
    if code or not count:
        raise RuntimeError(f"ffmpeg wrote no video ({'no frames' if not count else f'exit {code}'}): {error[-800:]}")
    return count


def render_frames(run, *, steps=None, size=(1280, 720), rotors=True, fuselage=True, label="", mode=None):
    """Yield one RGB image (H x W x 3, uint8) per stored frame."""
    import pyvista as pv
    from .scene import Scene
    plotter = pv.Plotter(off_screen=True, window_size=tuple(int(v) for v in size))
    try:
        scene = Scene(plotter, rotors=rotors, fuselage=fuselage)
        if mode is not None:
            scene.mode = mode
        for step in (run.frame_steps if steps is None else steps):
            frame = run.frame(step)
            name = str(frame["mode"]) + "   " if "mode" in frame else ""
            scene.draw(frame, label=label + name)
            yield np.asarray(plotter.screenshot(return_img=True))
    finally:
        plotter.close()


MAX_GIF_FRAMES = 600          # review STR-06: a GIF keeps its frames in memory (Pillow); above it every n-th stored frame


def gif_steps(run, steps=None):
    """The stored frames a GIF takes: all of them up to MAX_GIF_FRAMES, else every n-th (n the smallest that fits)."""
    steps = list(run.frame_steps if steps is None else steps)
    stride = max(1, -(-len(steps)//MAX_GIF_FRAMES))
    return steps[::stride], stride


def export_video(run, out_path, *, fps=12, steps=None, size=(1280, 720), **kwargs):
    """Write the video; the format follows the suffix: '.gif' (Pillow) or '.mp4' (imageio + ffmpeg). Returns the path.

    Review STR-06: MP4 is written frame by frame; a GIF holds its frames (Pillow writes them at the end), each reduced to
    a palette image as it is drawn (a third of the RGB size), at most MAX_GIF_FRAMES of them (every n-th stored frame
    above; the stride is in the file's comment). The renderer is closed whatever happens."""
    out = Path(out_path)
    if out.suffix.lower() == ".mp4":
        executable = ffmpeg_executable()
        if executable is not None:                     # VID-01: the ffmpeg on the PATH, frame by frame
            images = render_frames(run, steps=steps, size=size, **kwargs)
            try:
                write_mp4_ffmpeg(images, out, size, fps, executable)
            finally:
                images.close()
            return out
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise RuntimeError("MP4 needs ffmpeg on the PATH or imageio with its ffmpeg plug-in; write a .gif instead") from error
        images = render_frames(run, steps=steps, size=size, **kwargs)
        try:
            with imageio.get_writer(out, fps=fps) as writer:
                for image in images:
                    writer.append_data(image)
        finally:
            images.close()
        return out
    from PIL import Image
    chosen, stride = gif_steps(run, steps)
    if not chosen:
        raise ValueError("the run has no stored frames")
    images = render_frames(run, steps=chosen, size=size, **kwargs)
    try:
        frames = [Image.fromarray(image).quantize(colors=256, method=Image.Quantize.MEDIANCUT) for image in images]
    finally:
        images.close()
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(round(1000.0/fps)), loop=0, optimize=False,
                   comment=f"OpenFAWE Studio: every {stride} stored frame(s) of {run.path.name}".encode("utf-8"))
    return out
