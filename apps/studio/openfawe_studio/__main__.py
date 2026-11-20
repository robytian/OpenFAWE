"""python -m openfawe_studio [--runs FOLDER] [--smoke SECONDS --shot PNG] [--self-test [FOLDER]]"""
import argparse
import multiprocessing as mp
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openfawe_studio")
    parser.add_argument("--runs", default=None, help="folder for the runs (default: ~/OpenFAWE Studio runs)")
    parser.add_argument("--smoke", type=float, default=0.0, help="test: start the default run, wait, screenshot, quit")
    parser.add_argument("--shot", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--rotors", action="store_true")
    parser.add_argument("--motion", default=None)
    parser.add_argument("--wind", type=float, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--self-test", nargs="?", const="", default=None, metavar="FOLDER",
                        help="check the installation (OF-0013 RS-2): bundled cases, a 20-period flight, exports, the GPU")
    parser.add_argument("--write-selftest-reference", nargs=3, default=None, metavar=("OUT_JSON", "RUN_DIR", "BUILD"),
                        help=argparse.SUPPRESS)          # the packaging job, from source (OF-0013 addendum 4)
    args = parser.parse_args(argv)
    if args.self_test is not None:
        from . import selftest
        return selftest.main(args.self_test or None)
    if args.write_selftest_reference:
        from . import selftest
        out, run_dir, build = args.write_selftest_reference
        value = selftest.write_reference(out, run_dir, build)
        print(dict(periods=value["periods"], sha256=value["sha256"]))
        return 0 if value["periods"] == selftest.PERIODS else 1
    from PySide6 import QtCore, QtWidgets
    from .main_window import StudioWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = StudioWindow(args.runs)
    window.show()
    if args.smoke:
        import faulthandler
        faulthandler.dump_traceback_later(args.smoke+40, exit=True)     # a hung test prints where it hangs
        if args.steps:
            window.setup.widgets["steps"].setValue(args.steps)
        if args.rotors:
            window.setup.widgets["rotors"].setChecked(True)
        if args.motion:
            window.setup.widgets["motion"].setCurrentIndex(window.setup.widgets["motion"].findData(args.motion))
        if args.periods:
            window.setup.widgets["max_periods"].setValue(args.periods)
        if args.wind:
            window.setup.widgets["wind_m_s"].setValue(args.wind)
        QtCore.QTimer.singleShot(500, window.start)

        def done():
            if args.shot:
                if args.view:
                    window.scene.view(args.view)
                window.grab().save(args.shot)
                window.view3d.screenshot(args.shot.replace('.png', '-3d.png'))
            window.close()
            app.quit()
        QtCore.QTimer.singleShot(int(args.smoke*1000), done)
    return app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
