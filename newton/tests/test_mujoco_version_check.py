# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import types
import unittest
import warnings
from unittest import mock

import numpy as np

from newton._src.solvers.mujoco import solver_mujoco
from newton.exceptions import NewtonWarning

_MOCK_REQUIREMENTS = (
    "mujoco~=3.8.0 ; extra == 'sim'",
    "mujoco-warp~=3.8.0,>=3.8.0.3 ; extra == 'sim'",
)
_MOCK_METADATA = "\n".join(f"Requires-Dist: {requirement}" for requirement in _MOCK_REQUIREMENTS)


def _mujoco_dependency_specs():
    return {
        package: solver_mujoco._required_specifier(package, _MOCK_REQUIREMENTS) for package in ("mujoco", "mujoco-warp")
    }


class TestMuJoCoVersionCheck(unittest.TestCase):
    def setUp(self):
        mock_dist = types.SimpleNamespace(read_text=lambda name: _MOCK_METADATA)
        patcher = mock.patch.object(solver_mujoco.importlib_metadata, "distribution", return_value=mock_dist)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_warns_when_installed_versions_do_not_satisfy_pyproject(self):
        specs = _mujoco_dependency_specs()
        versions = {
            package: "0.0.0"
            for package, specifier in specs.items()
            if specifier and not solver_mujoco._version_satisfies("0.0.0", specifier)
        }

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with self.assertWarnsRegex(
                NewtonWarning,
                r"MuJoCo dependency version mismatch.*mujoco==0\.0\.0.*mujoco-warp==0\.0\.0",
            ):
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

    def test_warns_when_only_mujoco_warp_mismatches_pyproject(self):
        specs = _mujoco_dependency_specs()
        mujoco_warp_bad_version = "0.0.0"
        self.assertFalse(solver_mujoco._version_satisfies(mujoco_warp_bad_version, specs["mujoco-warp"]))

        versions = {"mujoco": _matching_version(specs["mujoco"]), "mujoco-warp": mujoco_warp_bad_version}
        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with self.assertWarnsRegex(NewtonWarning, f"mujoco-warp=={mujoco_warp_bad_version}"):
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

    def test_import_mujoco_warns_for_cached_mismatched_versions(self):
        specs = _mujoco_dependency_specs()
        versions = {
            package: "0.0.0"
            for package, specifier in specs.items()
            if specifier and not solver_mujoco._version_satisfies("0.0.0", specifier)
        }
        previous_mujoco = solver_mujoco.SolverMuJoCo._mujoco
        previous_mujoco_warp = solver_mujoco.SolverMuJoCo._mujoco_warp
        previous_versions_checked = solver_mujoco.SolverMuJoCo._versions_checked

        try:
            solver_mujoco.SolverMuJoCo._mujoco = types.SimpleNamespace()
            solver_mujoco.SolverMuJoCo._mujoco_warp = types.SimpleNamespace()
            solver_mujoco.SolverMuJoCo._versions_checked = False

            with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
                with self.assertWarnsRegex(NewtonWarning, "MuJoCo dependency version mismatch"):
                    solver_mujoco.SolverMuJoCo.import_mujoco()
        finally:
            solver_mujoco.SolverMuJoCo._mujoco = previous_mujoco
            solver_mujoco.SolverMuJoCo._mujoco_warp = previous_mujoco_warp
            solver_mujoco.SolverMuJoCo._versions_checked = previous_versions_checked

    def test_accepts_versions_that_satisfy_pyproject(self):
        versions = {
            package: _matching_version(specifier)
            for package, specifier in _mujoco_dependency_specs().items()
            if specifier
        }

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

        messages = [str(warning.message) for warning in caught]
        self.assertFalse(any("MuJoCo dependency version mismatch" in message for message in messages))

    def test_required_specifier_returns_none(self):
        cases = {
            "empty requirements": [],
            "package not in requirements": ["warp-lang>=1.0"],
        }
        for name, requirements in cases.items():
            with self.subTest(name):
                self.assertIsNone(solver_mujoco._required_specifier("mujoco", requirements))


class TestMuJoCoDeterminismConfig(unittest.TestCase):
    def test_modules_receive_their_record_bounds(self):
        """Give the smooth and solver modules their own record bounds and other modules 0."""
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver._deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
        solver._deterministic_max_records = 17
        solver._deterministic_solver_max_records = 3
        smooth_module = types.SimpleNamespace(__name__="mujoco_warp._src.smooth")
        solver_module = types.SimpleNamespace(__name__="mujoco_warp._src.solver")
        forward_module = types.SimpleNamespace(__name__="mujoco_warp._src.forward")

        with (
            mock.patch.object(
                solver_mujoco,
                "_mujoco_warp_deterministic_modules",
                return_value=[smooth_module, solver_module, forward_module],
            ),
            mock.patch.object(solver, "_set_module_options") as set_module_options,
        ):
            solver._set_mujoco_warp_module_options()

        dynamic_options = {
            "deterministic": solver_mujoco.wp.DeterministicMode.RUN_TO_RUN,
            "deterministic_max_records": 17,
        }
        solver_options = {
            "deterministic": solver_mujoco.wp.DeterministicMode.RUN_TO_RUN,
            "deterministic_max_records": 3,
        }
        generated_options = {
            "deterministic": solver_mujoco.wp.DeterministicMode.RUN_TO_RUN,
            "deterministic_max_records": 0,
        }
        self.assertEqual(
            set_module_options.call_args_list,
            [
                mock.call(dynamic_options, module=smooth_module),
                mock.call(solver_options, module=solver_module),
                mock.call(generated_options, module=forward_module),
                mock.call(generated_options, module=solver_mujoco.kernels),
                mock.call(
                    generated_options,
                    module=solver_mujoco.convert_newton_contacts_to_mjwarp_kernel.module,
                ),
            ],
        )

    @unittest.skipUnless(importlib.util.find_spec("mujoco_warp"), "mujoco_warp is not installed")
    def test_solver_record_bound_matches_dense_elliptic_launch(self):
        """Match the solver record bound to MJWarp's dense elliptic Hessian kernel launch."""
        from mujoco_warp import ConeType, EnableBit, IntegratorType, SolverType

        def bound(*, is_cuda=True, cone=ConeType.ELLIPTIC, solver=SolverType.NEWTON, is_sparse=False, ntri=300, **opt):
            options = {"enableflags": 0, "integrator": IntegratorType.EULER, **opt}
            mjw_model = types.SimpleNamespace(
                opt=types.SimpleNamespace(cone=cone, solver=solver, **options),
                is_sparse=is_sparse,
                dof_tri_row=np.zeros(ntri, dtype=np.int32),
            )
            mjw_data = types.SimpleNamespace(naconmax=1000, cdof_tri_row=np.zeros(32 * 32, dtype=np.int32))
            device = types.SimpleNamespace(is_cuda=is_cuda, sm_count=20)
            return solver_mujoco._mujoco_warp_solver_deterministic_max_records(mjw_model, mjw_data, device)

        # nv = 24 gives 300 DOF pairs, so the launch uses ceil(20 * 1536 / 300) = 103 blocks.
        self.assertEqual(bound(), math.ceil(1000 / 103))
        # Sleeping runs the compact solve over 32 * 32 DOF pairs, so the launch uses 30 blocks.
        self.assertEqual(bound(enableflags=EnableBit.SLEEP), math.ceil(1000 / 30))
        self.assertEqual(bound(enableflags=EnableBit.SLEEP, integrator=IntegratorType.DISCRETE), math.ceil(1000 / 103))
        for name, kwargs in {
            "cpu": {"is_cuda": False},
            "pyramidal": {"cone": ConeType.PYRAMIDAL},
            "cg": {"solver": SolverType.CG},
            "sparse": {"is_sparse": True},
            "sparse_sleeping": {"is_sparse": True, "enableflags": EnableBit.SLEEP},
            "no_dofs": {"ntri": 0},
        }.items():
            with self.subTest(name):
                self.assertEqual(bound(**kwargs), 0)

    def test_scoped_config_uses_solver_mode_and_restores_globals(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver._deterministic = solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED
        solver._deterministic_max_records = 0

        original_mode = solver_mujoco.wp.config.deterministic
        original_max_records = solver_mujoco.wp.config.deterministic_max_records
        try:
            solver_mujoco.wp.config.deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
            solver_mujoco.wp.config.deterministic_max_records = 17
            with solver._scoped_deterministic_config():
                self.assertEqual(
                    solver_mujoco.wp.config.deterministic,
                    solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED,
                )
                self.assertEqual(solver_mujoco.wp.config.deterministic_max_records, 0)

            self.assertEqual(solver_mujoco.wp.config.deterministic, solver_mujoco.wp.DeterministicMode.RUN_TO_RUN)
            self.assertEqual(solver_mujoco.wp.config.deterministic_max_records, 17)
        finally:
            solver_mujoco.wp.config.deterministic = original_mode
            solver_mujoco.wp.config.deterministic_max_records = original_max_records

    def test_notify_model_changed_uses_solver_config_for_mjwarp(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver.use_mujoco_cpu = False
        solver._deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
        solver._deterministic_max_records = 17
        solver.has_connect_constraints = False
        solver.has_jnt_connect_constraints = False
        observed_options = []

        def observe_options():
            observed_options.append(
                (
                    solver_mujoco.wp.config.deterministic,
                    solver_mujoco.wp.config.deterministic_max_records,
                )
            )

        original_mode = solver_mujoco.wp.config.deterministic
        original_max_records = solver_mujoco.wp.config.deterministic_max_records
        try:
            solver_mujoco.wp.config.deterministic = solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED
            solver_mujoco.wp.config.deterministic_max_records = 0
            with (
                mock.patch.object(solver, "_apply_module_options") as apply_module_options,
                mock.patch.object(solver, "_prepare_generated_kernels") as prepare_generated_kernels,
                mock.patch.object(solver, "_update_model_properties", side_effect=observe_options),
                mock.patch.object(solver, "_invalidate_contact_fast_path"),
            ):
                solver.notify_model_changed(solver_mujoco.ModelFlags.MODEL_PROPERTIES)

            apply_module_options.assert_called_once_with()
            prepare_generated_kernels.assert_called_once_with()
            self.assertEqual(
                observed_options,
                [(solver_mujoco.wp.DeterministicMode.RUN_TO_RUN, 17)],
            )
        finally:
            solver_mujoco.wp.config.deterministic = original_mode
            solver_mujoco.wp.config.deterministic_max_records = original_max_records

    def test_max_records_are_derived_from_model_dimensions(self):
        def make_model(nv, *, nu=0, tendon_num=(), ten_j_rownnz=(), **overrides):
            fields = {
                "nv": nv,
                "nu": nu,
                "nbody": 1,
                "body_weldid": (0,),
                "body_dofadr": (-1,),
                "body_dofnum": (0,),
                "dof_parentid": (),
                "tendon_num": tendon_num,
                "ten_J_rownnz": ten_j_rownnz,
                "flexedge_J_rownnz": (),
            }
            fields.update(overrides)
            return types.SimpleNamespace(**fields)

        independent_free_bodies = make_model(
            18,
            nbody=4,
            body_weldid=(0, 1, 2, 3),
            body_dofadr=(-1, 0, 6, 12),
            body_dofnum=(0, 6, 6, 6),
            dof_parentid=(-1, 0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10, -1, 12, 13, 14, 15, 16),
        )
        cases = {
            "constraint rows": (make_model(4), 200, 200),
            "sparse actuator Hessian": (make_model(32, nu=1), 64, 528),
            "spatial tendon": (make_model(8, tendon_num=(6,), ten_j_rownnz=(3,)), 2, 30),
            "fixed tendon armature": (make_model(1, tendon_num=(1,), ten_j_rownnz=(1,)), 0, 2),
            "independent body chains": (independent_free_bodies, 50, 78),
        }
        for name, (mj_model, njmax, expected) in cases.items():
            with self.subTest(name=name):
                mjw_data = types.SimpleNamespace(njmax=njmax)
                self.assertEqual(
                    solver_mujoco._mujoco_warp_deterministic_max_records(mj_model, mjw_data),
                    expected,
                )


def _matching_version(specifier: str) -> str:
    for pattern in (r">=\s*([0-9][^,;]*)", r"~=\s*([0-9][^,;]*)"):
        match = solver_mujoco.re.search(pattern, specifier)
        if match:
            return match.group(1)
    raise ValueError(f"_matching_version cannot derive a satisfying version from specifier {specifier!r}")


if __name__ == "__main__":
    unittest.main()
