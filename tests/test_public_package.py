from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from paper.settings import e1_grids, levels, option_strikes, probability_eval_paths, reserve_eval_paths, scale, seeds
from reproduce import TARGETS


ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_execution_profiles(self) -> None:
        self.assertEqual(scale("paper").replicates, 5)
        self.assertEqual(scale("paper").delta_paths, 20_000)
        self.assertEqual(scale("smoke").steps, 16)
        self.assertEqual(seeds("paper", 123), [123, 124, 125, 126, 127])
        self.assertEqual(levels("paper", appendix=True), [2, 3, 4, 5, 6])
        self.assertEqual(option_strikes("paper"), [95, 100, 105])
        self.assertEqual(probability_eval_paths("paper"), 3_000_000)
        self.assertEqual(reserve_eval_paths("paper"), 5_000_000)
        self.assertEqual(set(e1_grids("paper")), {
            "gbm_drift", "gbm_volatility", "merton_jump_intensity",
            "heston_vol_of_vol", "heston_self_sigma_v", "cev_beta",
        })

    def test_every_paper_target_is_exposed(self) -> None:
        self.assertTrue({"budget", "fixed", "options", "reserve", "bm19", "e1", "e2", "e3", "all"} <= set(TARGETS))

    def test_reference_inventory(self) -> None:
        tables = sorted((ROOT / "results/reference/tables").glob("*.csv"))
        expected_rows = {
            "01_budget_calibration_decoupled_vs_joint.csv": 4,
            "02_linear_risk_factor_expectation.csv": 4,
            "03_linear_tail_probability.csv": 12,
            "04_path_option_vol_sweep.csv": 12,
            "06_reserve_risk_family.csv": 24,
            "07_bm19_tailored_comparison.csv": 5,
        }
        self.assertEqual([path.name for path in tables], list(expected_rows))
        self.assertEqual(len(list((ROOT / "results/reference/appendix_e/e1").glob("*.csv"))), 3)
        self.assertEqual(len(list((ROOT / "results/reference/appendix_e/e2").glob("*.csv"))), 13)
        self.assertEqual(len(list((ROOT / "results/reference/appendix_e/e3").glob("*.csv"))), 2)
        self.assertEqual(len(list((ROOT / "results/reference/figures").glob("*.png"))), 4)
        for path in tables:
            frame = pd.read_csv(path)
            self.assertFalse(frame.empty, path.name)
            self.assertEqual(len(frame), expected_rows[path.name])
        appendix_rows = {
            "appendix_e/e1/appendix_e1_parametric_delta.csv": 330,
            "appendix_e/e1/appendix_e1_parametric_delta_k4.csv": 66,
            "appendix_e/e1/appendix_e1_parametric_delta_normalized.csv": 330,
            "appendix_e/e2/appendix_e2_pairwise_radii_long.csv": 90,
            "appendix_e/e3/appendix_e3_alpha_ablation.csv": 24,
            "appendix_e/e3/appendix_e3_signature_depth_ablation.csv": 30,
        }
        for relative, rows in appendix_rows.items():
            self.assertEqual(len(pd.read_csv(ROOT / "results/reference" / relative)), rows)

    def test_source_tree_is_self_contained(self) -> None:
        self.assertFalse((ROOT / "sig_dro/artifacts.py").exists())
        self.assertFalse((ROOT / "sig_dro/reserve_runner_utils.py").exists())
        self.assertIn("Table 6", TARGETS["reserve"])
        self.assertIn("Table 7", TARGETS["bm19"])
        sources = [ROOT / "reproduce.py", *sorted((ROOT / "paper").glob("*.py")),
                   *sorted((ROOT / "sig_dro").rglob("*.py"))]
        text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        self.assertNotIn("artifact_dir", text)
        self.assertNotIn("_engineered", text)
        self.assertNotIn("nn_seed", text)


if __name__ == "__main__":
    unittest.main()
