"""
Tests unitaires de base du jumeau numérique.

Lancer avec :  python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digital_twin.materials import get_alloy, ALLOY_DATABASE, AluminumAlloy, register_alloy
from digital_twin.thermal import ThermalModel
from digital_twin.electromagnetic import InductionCoilModel
from digital_twin.power_electronics import PowerElectronicsModel
from digital_twin.cooling import CoolingSystem
from digital_twin.sensors import make_standard_sensor_suite, SensorFaultMode
from digital_twin.aging import AgingModel
from digital_twin.faults import FaultInjector, FaultType, ScheduledFault
from digital_twin.simulation_engine import FurnaceDigitalTwin, FurnaceConfig


class TestMaterials(unittest.TestCase):
    def test_default_alloys_present(self):
        self.assertIn("AL1050", ALLOY_DATABASE)
        self.assertIn("AL6061", ALLOY_DATABASE)
        self.assertIn("AL380", ALLOY_DATABASE)

    def test_properties_increase_with_temperature_for_resistivity(self):
        alloy = get_alloy("AL1050")
        self.assertLess(alloy.resistivity(20), alloy.resistivity(600))

    def test_melt_fraction_bounds(self):
        alloy = get_alloy("AL1050")
        self.assertEqual(alloy.melt_fraction(alloy.solidus_c - 10), 0.0)
        self.assertEqual(alloy.melt_fraction(alloy.liquidus_c + 10), 1.0)
        mid = (alloy.solidus_c + alloy.liquidus_c) / 2
        self.assertTrue(0 < alloy.melt_fraction(mid) < 1)

    def test_register_new_alloy(self):
        custom = AluminumAlloy(
            name="TEST_ALLOY", solidus_c=600, liquidus_c=650,
            latent_heat_fusion=390000, density_solid=2700, density_liquid=2380,
            specific_heat_points={20: 900, 650: 1150},
            resistivity_points={20: 3e-8, 650: 2.4e-7},
            thermal_conductivity_points={20: 180, 650: 100},
            emissivity_points={20: 0.1, 650: 0.3},
            viscosity_liquid_points={650: 0.0013},
        )
        register_alloy(custom)
        self.assertIs(get_alloy("TEST_ALLOY"), custom)


class TestThermal(unittest.TestCase):
    def test_temperature_rises_with_positive_net_power(self):
        alloy = get_alloy("AL1050")
        model = ThermalModel(alloy=alloy, mass_kg=100, initial_temp_c=25)
        state = model.step(power_in_w=200_000, dt_s=10)
        self.assertGreater(state.temperature_c, 25)

    def test_temperature_falls_with_zero_power(self):
        alloy = get_alloy("AL1050")
        model = ThermalModel(alloy=alloy, mass_kg=100, initial_temp_c=500, ambient_temp_c=25)
        state = model.step(power_in_w=0, dt_s=10)
        self.assertLess(state.temperature_c, 500)

    def test_latent_heat_slows_temperature_rise_at_solidus(self):
        alloy = get_alloy("AL1050")
        model = ThermalModel(alloy=alloy, mass_kg=50, initial_temp_c=alloy.solidus_c - 0.5)
        state = model.step(power_in_w=50_000, dt_s=1)
        # La température ne doit pas dépasser drastiquement le solidus en un seul pas
        self.assertLessEqual(state.temperature_c, alloy.liquidus_c)


class TestElectromagnetic(unittest.TestCase):
    def test_skin_depth_decreases_with_frequency(self):
        d_low = InductionCoilModel.skin_depth(2.65e-8, 100, 1.0)
        d_high = InductionCoilModel.skin_depth(2.65e-8, 5000, 1.0)
        self.assertGreater(d_low, d_high)

    def test_coupling_efficiency_between_0_and_1(self):
        alloy = get_alloy("AL1050")
        coil = InductionCoilModel()
        state = coil.compute_state(alloy, 500, 1000)
        self.assertGreaterEqual(state.coupling_efficiency, 0.0)
        self.assertLessEqual(state.coupling_efficiency, 1.0)


class TestPowerElectronics(unittest.TestCase):
    def test_power_delivered_is_nonnegative(self):
        alloy = get_alloy("AL1050")
        coil = InductionCoilModel()
        em_state = coil.compute_state(alloy, 500, 1000)
        pe = PowerElectronicsModel()
        state = pe.compute_state(target_power_w=100_000, em_state=em_state)
        self.assertGreaterEqual(state.active_power_w, 0)
        self.assertGreaterEqual(state.power_factor, 0.0)
        self.assertLessEqual(state.power_factor, 1.0001)


class TestCooling(unittest.TestCase):
    def test_reduced_flow_reduces_capacity(self):
        cooling = CoolingSystem()
        full = cooling.step(heat_to_remove_w=10000, flow_fraction=1.0)
        half = cooling.step(heat_to_remove_w=10000, flow_fraction=0.5)
        self.assertGreater(full.cooling_capacity_w, half.cooling_capacity_w)

    def test_insufficient_flag(self):
        cooling = CoolingSystem(max_cooling_capacity_w=1000)
        state = cooling.step(heat_to_remove_w=5000, flow_fraction=1.0)
        self.assertFalse(state.is_sufficient)


class TestSensors(unittest.TestCase):
    def test_noise_applied(self):
        sensors = make_standard_sensor_suite()
        sensor = sensors["thermocouple_metal"]
        readings = [sensor.read(500.0, 1.0) for _ in range(20)]
        self.assertTrue(any(r != 500.0 for r in readings))

    def test_disconnected_returns_none(self):
        sensors = make_standard_sensor_suite()
        sensor = sensors["current_sensor"]
        sensor.set_fault(SensorFaultMode.DISCONNECTED)
        self.assertIsNone(sensor.read(100.0, 1.0))


class TestAging(unittest.TestCase):
    def test_health_decreases_over_time(self):
        aging = AgingModel()
        h0 = aging.health.igbt
        for _ in range(100):
            aging.step(
                dt_s=60, metal_temp_c=700, igbt_temp_c=120,
                capacitor_stress_factor=0.5, coil_current_a=900,
                coil_current_nominal_a=800, is_heating=True,
            )
        self.assertLess(aging.health.igbt, h0)


class TestFaults(unittest.TestCase):
    def test_scheduled_fault_triggers(self):
        injector = FaultInjector(random_fault_probability_per_hour=0.0)
        injector.schedule(ScheduledFault(FaultType.COOLING_LOSS, start_time_s=10, duration_s=20))
        injector.step(0, 10)
        active = injector.step(10, 1)
        self.assertTrue(any(f.fault_type == FaultType.COOLING_LOSS for f in active))

    def test_fault_expires(self):
        injector = FaultInjector(random_fault_probability_per_hour=0.0)
        injector.schedule(ScheduledFault(FaultType.COOLING_LOSS, start_time_s=0, duration_s=5))
        injector.step(0, 1)
        active = injector.step(10, 1)
        self.assertEqual(len(active), 0)


class TestSimulationEngine(unittest.TestCase):
    def test_full_step_runs_without_error(self):
        config = FurnaceConfig(mass_kg=200, max_power_w=200_000, seed=1)
        twin = FurnaceDigitalTwin(config)
        record = twin.step(1.0)
        self.assertGreaterEqual(record.temperature_metal_c, 0)
        self.assertIn(record.phase, ("heating", "melting", "holding", "cooling"))

    def test_temperature_increases_over_many_steps(self):
        config = FurnaceConfig(mass_kg=100, max_power_w=300_000, seed=2, random_fault_probability_per_hour=0.0)
        twin = FurnaceDigitalTwin(config)
        first_temp = twin.thermal.state.temperature_c
        for _ in range(200):
            record = twin.step(2.0)
        self.assertGreater(record.temperature_metal_c, first_temp)


if __name__ == "__main__":
    unittest.main()
