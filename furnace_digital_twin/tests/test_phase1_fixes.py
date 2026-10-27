"""
Tests de non-régression — Phase 1 (audit correctif).

Sept tests minimaux couvrant chacune des anomalies corrigées :
puissance, température, masse, fusion, économie, dataset, épisode.

Lancer avec :  python -m unittest discover -s tests
"""

import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digital_twin.materials import get_alloy
from digital_twin.thermal import ThermalModel
from digital_twin.simulation_engine import FurnaceDigitalTwin, FurnaceConfig
from digital_twin.database_generator import DatasetGenerator


class TestPuissance(unittest.TestCase):
    """5 % <= power_pct <= 100 %"""

    def test_power_pct_clamped_within_bounds(self):
        twin = FurnaceDigitalTwin(FurnaceConfig(mass_kg=20, max_power_w=100_000, seed=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_low = twin.step(1.0, power_pct=0.0)
            r_high = twin.step(1.0, power_pct=5.0)
        self.assertGreaterEqual(r_low.power_pct, 0.05)
        self.assertLessEqual(r_high.power_pct, 1.00)


class TestTemperature(unittest.TestCase):
    """T_bain <= 720 °C, en toutes circonstances (y compris overshoot)."""

    def test_bath_temperature_never_exceeds_720(self):
        twin = FurnaceDigitalTwin(FurnaceConfig(mass_kg=5, max_power_w=500_000, seed=1))
        twin.thermal.state.temperature_c = 719.8
        for _ in range(100):
            r = twin.step(5.0, power_pct=1.0)
            self.assertLessEqual(r.temperature_metal_c, 720.0)


class TestMasse(unittest.TestCase):
    """1 kg <= m_alu_initial <= 100 kg, et rejet des masses invalides."""

    def test_exploration_mass_within_bounds(self):
        gen = DatasetGenerator(dt_s=5, max_steps_per_episode=5, seed=1, exploration=True)
        config = gen.sampler.sample_exploration()
        self.assertGreaterEqual(config.mass_kg, 1.0)
        self.assertLessEqual(config.mass_kg, 100.0)

    def test_invalid_mass_raises(self):
        with self.assertRaises(ValueError):
            FurnaceDigitalTwin(FurnaceConfig(mass_kg=0))
        with self.assertRaises(ValueError):
            FurnaceDigitalTwin(FurnaceConfig(mass_kg=-1))


class TestFusion(unittest.TestCase):
    """La fraction liquide doit progresser après le solidus, sans se
    réinitialiser tant que le bain reste dans la zone pâteuse."""

    def test_melt_fraction_progresses_without_reset(self):
        alloy = get_alloy("AL1050")
        model = ThermalModel(alloy=alloy, mass_kg=10, initial_temp_c=644.0)
        fractions = []
        for _ in range(150):
            state = model.step(power_in_w=50_000, dt_s=1.0)
            fractions.append(state.melt_fraction)
        # Une fois positive, la fraction ne doit jamais retomber à 0
        # (sauf en fin de fusion où elle atteint 1 et y reste).
        started_rising = False
        for f in fractions:
            if f > 0.0:
                started_rising = True
            if started_rising:
                self.assertGreater(f, 0.0)
        self.assertEqual(fractions[-1], 1.0)


class TestEconomie(unittest.TestCase):
    """cost_usd = energy_kwh * 0.25 + m_dross * 0.60"""

    def test_cost_formula(self):
        twin = FurnaceDigitalTwin(FurnaceConfig(mass_kg=20, max_power_w=100_000, seed=1))
        r = twin.step(1.0, power_pct=1.0)
        expected = r.energy_kwh * 0.25 + r.m_dross_kg * 0.60
        self.assertAlmostEqual(r.cost_usd, expected, places=9)

    def test_energy_based_on_active_power(self):
        twin = FurnaceDigitalTwin(FurnaceConfig(mass_kg=20, max_power_w=100_000, seed=1))
        r = twin.step(1.0, power_pct=1.0)
        self.assertAlmostEqual(r.energy_consumed_j, r.active_power_w * 1.0, places=6)


class TestDataset(unittest.TestCase):
    """Colonnes obligatoires du CSV en mode exploration."""

    REQUIRED_COLUMNS = (
        "timestamp", "T_bain", "power_pct", "P_effective",
        "m_alu", "m_dross", "energy_kwh", "cost_usd", "yield_pct",
    )

    def test_required_columns_present(self):
        gen = DatasetGenerator(dt_s=5, max_steps_per_episode=5, seed=1, exploration=True)
        rows = list(gen.run_episode(0))
        self.assertGreater(len(rows), 0)
        for col in self.REQUIRED_COLUMNS:
            self.assertIn(col, rows[0])

    def test_yield_pct_bounded(self):
        gen = DatasetGenerator(dt_s=5, max_steps_per_episode=30, seed=2, exploration=True)
        rows = list(gen.run_episode(0))
        for row in rows:
            self.assertGreaterEqual(row["yield_pct"], 0.0)
            self.assertLessEqual(row["yield_pct"], 100.0)


class TestEpisode(unittest.TestCase):
    """Un épisode d'exploration doit pouvoir réellement se terminer, et
    son statut de complétion doit être enregistré explicitement."""

    def test_episode_can_complete(self):
        # Petite masse + forte puissance + maintien court -> cycle rapide.
        config = FurnaceConfig(
            mass_kg=2.0, max_power_w=200_000, hold_duration_s=10.0,
            target_temp_c=720.0, seed=1, random_fault_probability_per_hour=0.0,
        )
        gen = DatasetGenerator(dt_s=2.0, max_steps_per_episode=5000, seed=1)
        rows = list(gen.run_episode(0, config=config))
        self.assertTrue(rows[-1]["completed"])
        self.assertIn("completed", rows[0])


if __name__ == "__main__":
    unittest.main()
