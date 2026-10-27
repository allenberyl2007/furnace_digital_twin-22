"""
Tests de non-régression — Audit qualité dataset (générateur = source de
vérité : chaque dataset produit doit être directement exploitable pour
l'entraînement d'une IA, sans nettoyage manuel).

Quatre tests minimaux couvrant chacune des garanties demandées :
- un dataset généré ne contient jamais d'épisode incomplet ;
- les colonnes critiques IA sont toujours renseignées (jamais NaN/inf),
  y compris hors mode exploration ;
- aucune valeur physiquement impossible ne peut être écrite (le
  générateur lève une erreur explicite plutôt que de la laisser passer) ;
- le nombre d'épisodes réellement produits correspond à la demande.

Lancer avec :  python -m unittest discover -s tests
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digital_twin.database_generator import DatasetGenerator, _validate_critical_columns


class TestDatasetToujoursComplet(unittest.TestCase):
    """`generate()` ne doit jamais inclure d'épisode incomplet : tout
    épisode qui n'atteint pas `is_finished()` avant `max_steps_per_episode`
    est écarté et remplacé, jusqu'à obtenir exactement `n_episodes`
    épisodes complets."""

    def test_generate_only_yields_completed_episodes(self):
        gen = DatasetGenerator(dt_s=1.0, max_steps_per_episode=20000, seed=7, exploration=True)
        rows = list(gen.generate(15))
        episode_ids = set()
        for row in rows:
            self.assertTrue(row["completed"])
            episode_ids.add(row["episode_id"])
        self.assertEqual(len(episode_ids), 15)
        self.assertEqual(episode_ids, set(range(15)))

    def test_generate_works_without_exploration_flag_too(self):
        # Reproduit le scénario du dataset industriel (mode par défaut,
        # sans --exploration) qui avait révélé le bug (0/5 épisodes
        # complets) : doit désormais aussi garantir des épisodes complets.
        gen = DatasetGenerator(dt_s=1.0, max_steps_per_episode=20000, seed=42, exploration=False)
        rows = list(gen.generate(3))
        for row in rows:
            self.assertTrue(row["completed"])


class TestColonnesCritiquesJamaisNaN(unittest.TestCase):
    """`power_pct` et les colonnes du contrôleur optionnel ne doivent
    jamais être NaN, y compris hors mode exploration et sans contrôleur
    optimal (c'est précisément le cas qui produisait un dataset avec ces
    colonnes 100 % NaN)."""

    def test_power_pct_never_nan_without_exploration_or_controller(self):
        gen = DatasetGenerator(dt_s=1.0, max_steps_per_episode=200, seed=3, exploration=False)
        rows = list(gen.run_episode(0))
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertFalse(math.isnan(row["power_pct"]))
            self.assertFalse(math.isnan(row["optimal_power_setpoint_w"]))
            self.assertFalse(math.isnan(row["optimal_cooling_flow_fraction"]))


class TestValidationValeursImpossibles(unittest.TestCase):
    """`_validate_critical_columns` doit rejeter explicitement toute
    valeur critique invalide plutôt que de la laisser passer."""

    def test_rejects_temperature_above_absolute_limit(self):
        with self.assertRaises(ValueError):
            _validate_critical_columns({"T_bain": 721.0})

    def test_rejects_negative_cost(self):
        with self.assertRaises(ValueError):
            _validate_critical_columns({"cost_usd": -0.5})

    def test_rejects_yield_out_of_bounds(self):
        with self.assertRaises(ValueError):
            _validate_critical_columns({"yield_pct": 105.0})

    def test_accepts_valid_row(self):
        try:
            _validate_critical_columns({
                "T_bain": 700.0, "power_pct": 0.5, "P_effective": 1000.0,
                "m_alu": 10.0, "m_dross": 0.1, "energy_kwh": 1.0,
                "cost_usd": 0.5, "yield_pct": 99.0, "mass_kg": 10.1,
            })
        except ValueError:
            self.fail("_validate_critical_columns a rejeté une ligne valide.")


class TestSchemaEtVolumeStable(unittest.TestCase):
    """`export_csv` doit produire exactement `n_episodes` épisodes, tous
    avec le même schéma de colonnes (aucune colonne n'apparaît/disparaît
    d'une ligne à l'autre)."""

    def test_export_csv_row_and_episode_count(self):
        gen = DatasetGenerator(dt_s=2.0, max_steps_per_episode=2000, seed=9, exploration=True)
        output = gen.export_csv("/tmp/test_dataset_quality_export.csv", n_episodes=5)
        import csv
        with open(output) as f:
            rows = list(csv.DictReader(f))
        episode_ids = {row["episode_id"] for row in rows}
        self.assertEqual(len(episode_ids), 5)
        first_keys = set(rows[0].keys())
        for row in rows:
            self.assertEqual(set(row.keys()), first_keys)
        output.unlink()


class TestCiblesIAOntUneVariance(unittest.TestCase):
    """Audit qualité IA (§5/§9 du rapport d'audit) : `yield_pct` et
    `maintenance_alert` doivent présenter une variance réelle sur un
    dataset généré — sinon ils sont inutilisables comme cible d'IA. Ces
    deux tests figent le correctif de calibration (`DROSS_RATE_COEFFICIENT_KG_M2_S`
    et multiplicateur d'usure du réfractaire) : ils échoueraient si l'un
    des deux redevenait quasi constant par régression."""

    def test_yield_pct_has_meaningful_variance(self):
        gen = DatasetGenerator(dt_s=1.0, max_steps_per_episode=20000, seed=100, exploration=True)
        rows = list(gen.generate(20))
        finals = {}
        for row in rows:
            finals[row["episode_id"]] = row["yield_pct"]
        values = list(finals.values())
        # Avant correction : std ≈ 0.0003 (quasi constant à 99.9998%).
        # Après correction : std attendu de l'ordre de 1.0.
        std = (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5
        self.assertGreater(std, 0.1, "yield_pct est quasi constant : cible de régression inexploitable.")

    def test_maintenance_alert_is_reachable(self):
        # Avec le multiplicateur d'usure corrigé, `maintenance_alert=1`
        # doit pouvoir être observé sur un échantillon d'épisodes de
        # taille raisonnable (avant correction : 0 déclenchement même sur
        # 500 épisodes, seuil structurellement inatteignable).
        gen = DatasetGenerator(dt_s=1.0, max_steps_per_episode=20000, seed=21, exploration=True)
        rows = list(gen.generate(300))
        self.assertTrue(
            any(row["maintenance_alert"] == 1 for row in rows),
            "maintenance_alert ne se déclenche jamais sur 300 épisodes : seuil inatteignable.",
        )


if __name__ == "__main__":
    unittest.main()
