"""Tests of the published aerodynamic module.

Four tests of the development repository compare with internal records (retained earlier sources and probes) that are
not part of this release; they are deselected here.
"""
INTERNAL_RECORDS = {
    "tests/test_camber.py::CamberTests::test_thin_airfoil_integrals_and_profile_datum",
    "tests/test_initial_cap.py::InitialCapTests::test_legacy_initial_arrays_match_sealed_startup_records",
    "tests/test_vsm_core_profiles.py::CoreProfileTests::test_legacy_and_none_numeric_and_metadata_bridge",
    "tests/test_vsm_core_profiles.py::CoreProfileTests::test_native_six_point_oracle",
}


def pytest_collection_modifyitems(config, items):
    keep = [item for item in items if item.nodeid not in INTERNAL_RECORDS]
    if len(keep) != len(items):
        config.hook.pytest_deselected(items=[item for item in items if item.nodeid in INTERNAL_RECORDS])
        items[:] = keep
