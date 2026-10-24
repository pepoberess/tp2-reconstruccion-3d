"""Pruebas del registro con una transformación conocida, sin fotos ni modelos."""
import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import fusion
import pose


class ICPTests(unittest.TestCase):
    """Prueba el refinamiento ICP y las utilidades relacionadas."""

    @classmethod
    def setUpClass(cls):
        """Prepara una nube sintética y su transformación conocida."""
        rng = np.random.default_rng(42)
        directions = rng.normal(size=(6000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        cls.points = directions * [40, 27, 19] + [240, -300, 80]
        cls.target = fusion.make_pcd(cls.points)
        cls.error = np.eye(4)
        cls.error[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.006, -0.008, 0.01])
        cls.error[:3, 3] = [0.6, -0.4, 0.5]
        cls.source = fusion.make_pcd(pose.transform_points(cls.points, cls.error))

    def test_known_transform_and_pose_composition(self):
        """Verifica que ICP recupere una transformación pequeña conocida."""
        delta, info = fusion.refine_with_icp(self.source, self.target)
        self.assertTrue(info["accepted"], info)
        recovered = pose.transform_points(np.asarray(self.source.points), delta)
        self.assertLess(np.sqrt(np.mean((recovered - self.points) ** 2)), 0.03)
        np.testing.assert_allclose(delta @ self.error, np.eye(4), atol=0.03)
        self.assertLess(info["overlap_rmse_candidate_mm"], info["overlap_rmse_before_mm"])

    def test_large_correction_is_rejected(self):
        """Verifica el rechazo de una corrección mayor que el límite."""
        delta, info = fusion.refine_with_icp(self.source, self.target, {"max_shift_mm": 0.001})
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "corrección demasiado grande")
        np.testing.assert_array_equal(delta, np.eye(4))

    def test_no_overlap_and_empty_cloud(self):
        """Verifica el manejo de nubes vacías o sin solapamiento."""
        for source in [o3d.geometry.PointCloud(), fusion.make_pcd(self.points + 1000)]:
            delta, info = fusion.refine_with_icp(source, self.target)
            self.assertFalse(info["accepted"])
            np.testing.assert_array_equal(delta, np.eye(4))

    def test_board_is_excluded_without_mutating_input(self):
        """Verifica que el tablero se excluya sin modificar la nube original."""
        points = np.array([[2, 2, 0], [3, 3, 20], [5, 5, 35], [90, 90, 40]], float)
        cloud = fusion.make_pcd(points)
        cropped = fusion.prepare_icp_cloud(cloud, [0, 0, -20], [50, 50, 100], {})
        self.assertEqual(len(cropped.points), 1)
        np.testing.assert_array_equal(np.asarray(cloud.points), points)

    def test_relative_error(self):
        """Verifica el cálculo y la validación del error relativo."""
        self.assertAlmostEqual(fusion.relative_error(133.4, 140.0), 0.0471428571)
        self.assertAlmostEqual(fusion.relative_error(146.6, 140.0), 0.0471428571)
        with self.assertRaises(ValueError):
            fusion.relative_error(133.4, 0)


if __name__ == "__main__":
    unittest.main()
