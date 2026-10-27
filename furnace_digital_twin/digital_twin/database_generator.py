"""
Générateur de base de données
================================

Enchaîne plusieurs épisodes de simulation (chaque épisode = un cycle complet
de fusion d'une charge d'aluminium) avec des paramètres initiaux
aléatoires mais physiquement plausibles, et exporte les enregistrements de
télémétrie dans un fichier CSV prêt à l'emploi pour l'entraînement d'une IA
supervisée (détection d'anomalie, maintenance prédictive, régression de
rendement, etc.).
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable, Iterator

from .materials import ALLOY_DATABASE
from .simulation_engine import FurnaceDigitalTwin, FurnaceConfig, SimulationRecord, ProcessPhase
from .data_exporter import DataExporter

# Colonnes qui ne doivent jamais contenir NaN/inf/valeurs négatives
# impossibles dans le dataset d'exploration (valeurs calculées par le
# modèle, pas des mesures de capteur pouvant légitimement manquer).
_CRITICAL_EXPLORATION_COLUMNS = (
    "T_bain", "power_pct", "P_effective", "m_alu", "m_dross",
    "energy_kwh", "cost_usd", "yield_pct", "mass_kg",
)

# Colonnes explicitement réservées à des fonctionnalités optionnelles
# (contrôleur prédictif `OptimalController`, désactivé par défaut). Elles
# restent présentes pour la stabilité du schéma mais sont exclues de la
# validation "colonne IA critique" : voir le remplissage à 0.0 plus bas
# (`_fill_optional_controller_columns`), qui évite déjà qu'elles soient
# entièrement NaN.
_OPTIONAL_CONTROLLER_COLUMNS = ("optimal_power_setpoint_w", "optimal_cooling_flow_fraction")


def _validate_critical_columns(row: dict) -> None:
    """Vérification simple (pas de nettoyage automatique) : lève une
    erreur explicite si une colonne critique du dataset contient une
    valeur non exploitable, plutôt que de la laisser passer silencieusement.
    """
    for col in _CRITICAL_EXPLORATION_COLUMNS:
        value = row.get(col)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Valeur invalide ({value}) dans la colonne critique '{col}'.")
    # Bornes physiques impossibles (au-delà du simple NaN/inf).
    t_bain = row.get("T_bain")
    if t_bain is not None and t_bain > FurnaceDigitalTwin.ABSOLUTE_MAX_BATH_TEMP_C + 1e-6:
        raise ValueError(f"T_bain={t_bain} dépasse la limite de sécurité absolue (720 °C).")
    for col in ("cost_usd", "energy_kwh", "m_dross", "m_alu", "mass_kg"):
        value = row.get(col)
        if isinstance(value, (int, float)) and value < -1e-9:
            raise ValueError(f"Valeur négative impossible ({value}) dans la colonne '{col}'.")
    yield_pct = row.get("yield_pct")
    if isinstance(yield_pct, (int, float)) and not (-1e-9 <= yield_pct <= 100.0 + 1e-6):
        raise ValueError(f"yield_pct={yield_pct} hors bornes [0, 100].")
    power_pct = row.get("power_pct")
    # 0.0 est une valeur légitime lorsqu'aucune puissance n'a été
    # appliquée ce pas-ci (ex. phase de refroidissement après coulée) ;
    # seules les valeurs strictement comprises entre 0 et le plancher de
    # 5 % (hors 0 exact) seraient le signe d'un bug de bornage.
    if isinstance(power_pct, (int, float)) and not (
        power_pct == 0.0 or 0.05 - 1e-9 <= power_pct <= 1.0 + 1e-9
    ):
        raise ValueError(f"power_pct={power_pct} hors bornes {{0.0}} ∪ [0.05, 1.00].")


class ParameterSampler:
    """Tire des paramètres d'épisode aléatoires mais physiquement plausibles."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def sample(self) -> FurnaceConfig:
        alloy_name = self.rng.choice(list(ALLOY_DATABASE.keys()))
        mass_kg = self.rng.uniform(150.0, 1200.0)
        ambient_temp_c = self.rng.uniform(10.0, 40.0)
        # BUG CORRIGÉ (audit dataset — épisodes jamais terminés) : la borne
        # haute était 780.0°C, au-delà de la sécurité thermique absolue
        # (`FurnaceDigitalTwin.ABSOLUTE_MAX_BATH_TEMP_C` = 720°C). Un
        # objectif de phase MELTING->HOLDING physiquement inatteignable
        # (température plafonnée à 720°C bien avant la cible) bloquait la
        # simulation indéfiniment en phase MELTING, empêchant tout accès à
        # HOLDING puis COOLING -> l'épisode n'atteignait jamais
        # `is_finished()`. Bornée à 720.0°C, comme déjà fait dans
        # `sample_exploration()`.
        target_temp_c = self.rng.uniform(700.0, 720.0)
        max_power_w = self.rng.uniform(150_000.0, 900_000.0)
        frequency_hz = self.rng.uniform(150.0, 3000.0)
        grid_voltage_v = self.rng.choice([380.0, 400.0, 415.0, 480.0])
        hold_duration_s = self.rng.uniform(120.0, 1800.0)
        cooling_flow_lpm = self.rng.uniform(20.0, 60.0)
        cooling_inlet_temp_c = self.rng.uniform(12.0, 28.0)
        fault_probability = self.rng.uniform(0.0, 0.25)
        initial_wear_level = self.rng.betavariate(1.5, 4.0)  # majoritairement peu usé

        return FurnaceConfig(
            alloy_name=alloy_name,
            mass_kg=mass_kg,
            initial_temp_c=ambient_temp_c,
            ambient_temp_c=ambient_temp_c,
            target_temp_c=target_temp_c,
            hold_duration_s=hold_duration_s,
            max_power_w=max_power_w,
            frequency_hz=frequency_hz,
            grid_voltage_v=grid_voltage_v,
            cooling_nominal_flow_lpm=cooling_flow_lpm,
            cooling_inlet_temp_c=cooling_inlet_temp_c,
            random_fault_probability_per_hour=fault_probability,
            seed=self.rng.randrange(1, 10**9),
            initial_wear_level=initial_wear_level,
        )

    def sample_exploration(self) -> FurnaceConfig:
        """Variante de `sample()` pour le mode exploration (`--exploration`).

        Cible spécifiquement de petites charges (1 à 100 kg, typiques
        d'essais/lots réduits) plutôt que la plage industrielle par défaut
        de `sample()` (150 à 1200 kg), afin de couvrir la plage demandée
        pour les analyses de coût/rendement en fonction de `power_pct`.
        """
        alloy_name = self.rng.choice(list(ALLOY_DATABASE.keys()))
        mass_kg = self.rng.uniform(1.0, 100.0)
        ambient_temp_c = self.rng.uniform(10.0, 40.0)
        # La sécurité thermique absolue coupe à 720 °C : viser cette même
        # valeur comme cible évite des épisodes qui ne terminent jamais
        # leur phase de maintien faute de pouvoir dépasser le seuil.
        target_temp_c = 720.0
        # BUG CORRIGÉ (audit Phase 1, §11) : `exposed_surface_m2` restait
        # implicitement à sa valeur par défaut (0.6 m², calibrée pour
        # ~500 kg) quelle que soit la masse tirée dans [1, 100] kg. Pour
        # les petites masses, la surface libre relative devenait
        # disproportionnée et les pertes (rayonnement/convection)
        # empêchaient physiquement d'atteindre le solidus, quelle que soit
        # la puissance appliquée -> épisodes bloqués en phase 'heating'.
        # Mise à l'échelle géométrique simple (surface ~ masse^(2/3) à
        # forme constante), calibrée pour retrouver 0.6 m² à 500 kg.
        exposed_surface_m2 = 0.6 * (mass_kg / 500.0) ** (2.0 / 3.0)
        # BUG CORRIGÉ (idem) : la puissance nominale était tirée
        # indépendamment de la masse (20-300 kW fixe), si bien que les
        # masses élevées pouvaient se retrouver avec trop peu de
        # puissance pour jamais atteindre le solidus dans le budget de
        # pas disponible. Mise à l'échelle en kW/kg (quasi constante),
        # pour un temps de chauffe borné indépendamment de la masse.
        max_power_w = mass_kg * self.rng.uniform(3000.0, 8000.0)
        frequency_hz = self.rng.uniform(150.0, 3000.0)
        grid_voltage_v = self.rng.choice([380.0, 400.0, 415.0, 480.0])
        hold_duration_s = self.rng.uniform(60.0, 600.0)
        cooling_flow_lpm = self.rng.uniform(10.0, 40.0)
        cooling_inlet_temp_c = self.rng.uniform(12.0, 28.0)
        # CORRECTION (audit qualité IA) : la borne haute précédente (0.15
        # panne/heure) produisait, empiriquement, une quasi-absence de
        # pannes injectées (0 à ~6 % des épisodes avec au moins une panne
        # active sur plusieurs échantillons de 50 épisodes) — la colonne
        # `active_faults` était de ce fait presque inexploitable comme
        # signal d'entraînement (classe positive quasi vide). Relevée à
        # 5.0/heure : validé empiriquement (~40-50 % des épisodes ont
        # alors au moins une panne active sur un échantillon de 50),
        # tout en conservant une majorité d'épisodes sans panne (pas de
        # sur-représentation artificielle qui déséquilibrerait le
        # problème dans l'autre sens).
        fault_probability = self.rng.uniform(0.0, 5.0)
        initial_wear_level = self.rng.betavariate(1.5, 4.0)

        return FurnaceConfig(
            alloy_name=alloy_name,
            mass_kg=mass_kg,
            initial_temp_c=ambient_temp_c,
            ambient_temp_c=ambient_temp_c,
            target_temp_c=target_temp_c,
            hold_duration_s=hold_duration_s,
            max_power_w=max_power_w,
            frequency_hz=frequency_hz,
            grid_voltage_v=grid_voltage_v,
            exposed_surface_m2=exposed_surface_m2,
            cooling_nominal_flow_lpm=cooling_flow_lpm,
            cooling_inlet_temp_c=cooling_inlet_temp_c,
            random_fault_probability_per_hour=fault_probability,
            seed=self.rng.randrange(1, 10**9),
            initial_wear_level=initial_wear_level,
        )


class PowerProfile:
    """Profil de puissance (`power_pct`) appliqué au cours d'un épisode.

    Deux modes :
    - puissance constante (`is_constant=True`) : `power_pct` fixe sur tout
      l'épisode ;
    - paliers aléatoires (`is_constant=False`) : la valeur de `power_pct`
      change à des instants aléatoires, par palier, chaque palier tirant
      une nouvelle valeur uniforme entre 5 % et 100 %.
    """

    def __init__(
        self,
        rng: random.Random,
        is_constant: bool,
        constant_value: float | None = None,
        min_step_duration_s: float = 30.0,
        max_step_duration_s: float = 300.0,
    ) -> None:
        self.rng = rng
        self.is_constant = is_constant
        self.constant_value = constant_value if constant_value is not None else rng.uniform(0.05, 1.0)
        self.min_step_duration_s = min_step_duration_s
        self.max_step_duration_s = max_step_duration_s
        self._current_value = self.constant_value
        self._next_change_time_s = 0.0

    def value_at(self, time_s: float) -> float:
        """Retourne la valeur de `power_pct` à appliquer à l'instant `time_s`."""
        if self.is_constant:
            return self.constant_value
        if time_s >= self._next_change_time_s:
            self._current_value = self.rng.uniform(0.05, 1.0)
            self._next_change_time_s = time_s + self.rng.uniform(
                self.min_step_duration_s, self.max_step_duration_s
            )
        return self._current_value

    @classmethod
    def sample(cls, rng: random.Random) -> "PowerProfile":
        """Tire au hasard un profil constant ou par paliers (50/50)."""
        is_constant = rng.random() < 0.5
        return cls(rng=rng, is_constant=is_constant)


class DatasetGenerator:
    """Génère un jeu de données de télémétrie multi-épisodes."""

    def __init__(
        self,
        dt_s: float = 1.0,
        max_steps_per_episode: int = 20000,
        seed: int | None = None,
        with_optimal_control: bool = False,
        export_format: str = "csv",
        exploration: bool = False,
    ) -> None:
        """Initialise le générateur de dataset.

        Args:
            dt_s: Pas de temps de simulation (s).
            max_steps_per_episode: Nombre maximal de pas par épisode.
            seed: Graine aléatoire globale (paramètres + simulations).
            with_optimal_control: Si vrai, chaque épisode généré active
                `FurnaceConfig.use_optimal_control` (contrôleur prédictif
                au lieu de la logique de consigne historique). Faux par
                défaut : comportement inchangé si l'option n'est pas
                activée, pour rester rétrocompatible.
            export_format: Format d'export utilisé par `export_csv`
                ("csv" ou "sqlite"), transmis à `DataExporter`. Le nom de
                la méthode `export_csv` est conservé pour la
                rétrocompatibilité même lorsque `export_format="sqlite"`.
            exploration: Si vrai, active le mode d'exploration : masse
                initiale tirée dans [1, 100] kg (`ParameterSampler.sample_exploration`)
                et profil de puissance (`PowerProfile`) constant ou par
                paliers aléatoires entre 5 % et 100 %, appliqué via le
                paramètre `power_pct` de `FurnaceDigitalTwin.step()`.
                Faux par défaut : comportement inchangé (`sample()` +
                logique de consigne historique/contrôleur) si l'option
                n'est pas activée.
        """
        self.dt_s = dt_s
        self.max_steps_per_episode = max_steps_per_episode
        self.rng = random.Random(seed)
        self.sampler = ParameterSampler(self.rng)
        self.with_optimal_control = with_optimal_control
        self.export_format = export_format
        self.exploration = exploration

    def run_episode(self, episode_id: int, config: FurnaceConfig | None = None) -> Iterator[dict]:
        if config is None:
            config = self.sampler.sample_exploration() if self.exploration else self.sampler.sample()
        if self.with_optimal_control:
            config.use_optimal_control = True

        power_profile = PowerProfile.sample(self.rng) if self.exploration else None

        twin = FurnaceDigitalTwin(config)
        steps = 0
        # Les lignes sont tamponnées le temps de l'épisode (borné par
        # `max_steps_per_episode`, déjà la limite mémoire implicite du
        # système existant) afin de pouvoir leur ajouter après coup le
        # statut `completed`, connu seulement une fois l'épisode terminé.
        episode_rows: list[dict] = []
        while steps < self.max_steps_per_episode:
            # En phase COOLING, ne pas imposer le profil de puissance
            # exploration : sinon la consigne aléatoire (>= 5 % par
            # construction) continue d'injecter de la puissance alors que
            # le four est censé refroidir, et le bain ne redescend jamais
            # vers l'ambiante -> l'épisode n'atteint jamais `is_finished()`
            # et consomme systématiquement tout `max_steps_per_episode`.
            if power_profile is not None and twin.phase != ProcessPhase.COOLING:
                power_pct = power_profile.value_at(twin.time_s)
                record = twin.step(self.dt_s, power_pct=power_pct)
            else:
                record = twin.step(self.dt_s)
            row = asdict(record)
            row["episode_id"] = episode_id
            row["alloy"] = config.alloy_name
            row["mass_kg"] = config.mass_kg

            # BUG CORRIGÉ (audit qualité dataset) : `power_pct` et les deux
            # colonnes du contrôleur optionnel restaient NaN sur 100 % des
            # lignes dès que le pas courant n'était pas explicitement
            # piloté par `power_pct=...` (mode historique/COOLING) ou que
            # `use_optimal_control` était désactivé (cas par défaut). Une
            # colonne entièrement NaN casse la plupart des pipelines
            # d'entraînement IA. On la remplace donc par une valeur
            # numérique correctement renseignée plutôt que de la laisser
            # vide : la fraction de puissance réellement appliquée pour
            # `power_pct` (donnée calculée légitime, pas une mesure
            # manquante), et 0.0 pour les colonnes du contrôleur optionnel
            # (0.0 = contrôleur non sollicité ce pas-ci ; ces deux colonnes
            # ne contiennent des valeurs non nulles que sur les épisodes
            # générés avec `--with-optimal-control`, où elles restent
            # informatives).
            if math.isnan(row["power_pct"]):
                row["power_pct"] = (
                    row["p_effective_w"] / config.max_power_w if config.max_power_w > 0 else 0.0
                )
            for col in _OPTIONAL_CONTROLLER_COLUMNS:
                if math.isnan(row[col]):
                    row[col] = 0.0

            if self.exploration:
                # Colonnes explicitement demandées pour le mode
                # exploration (certaines dupliquent des champs déjà
                # présents dans `SimulationRecord` sous un autre nom, pour
                # correspondre exactement au schéma demandé).
                row["timestamp"] = record.time_s
                row["T_bain"] = record.temperature_metal_c
                # m_alu = masse d'aluminium RESTANTE (masse initiale moins
                # la calamine/dross formée), pas la masse initiale répétée
                # à chaque ligne.
                row["m_alu"] = max(0.0, config.mass_kg - record.m_dross_kg)
                row["m_dross"] = record.m_dross_kg
                row["P_effective"] = record.p_effective_w
                _validate_critical_columns(row)
            episode_rows.append(row)
            steps += 1
            if twin.is_finished():
                break

        # Un épisode qui atteint `max_steps_per_episode` sans que le four
        # ait terminé son cycle (retour proche de l'ambiante en phase
        # COOLING) ne doit jamais être présenté silencieusement comme un
        # cycle valide : le statut est enregistré explicitement sur
        # chaque ligne plutôt que de lever une exception qui interromprait
        # toute la génération du dataset (ex: --episodes 1000).
        completed = twin.is_finished()
        for row in episode_rows:
            row["completed"] = completed
            yield row

    def generate(self, n_episodes: int, max_attempts_multiplier: int = 5) -> Iterator[dict]:
        """Génère `n_episodes` épisodes *complets*.

        BUG CORRIGÉ (audit qualité dataset) : un épisode qui n'atteint pas
        `is_finished()` avant `max_steps_per_episode` était tout de même
        inclus dans le dataset final (simplement marqué
        `completed=False`), ce qui obligeait la personne entraînant une IA
        à filtrer elle-même le fichier avant usage — contraire à l'objectif
        "directement exploitable sans nettoyage manuel". Un épisode
        incomplet n'est donc plus jamais inclus : il est silencieusement
        écarté et remplacé par un nouvel épisode (nouveaux paramètres
        aléatoires), jusqu'à obtenir exactement `n_episodes` épisodes
        complets, avec des `episode_id` réattribués séquentiellement
        (0 à n_episodes-1) pour que le dataset final reste propre.

        Une seule ligne de chaque épisode (le dernier pas) suffit pour
        savoir s'il faut le garder ; le tampon d'un épisode reste borné
        par `max_steps_per_episode`, jamais par `n_episodes` : la mémoire
        ne croît donc pas avec le nombre d'épisodes demandés (déjà vérifié
        empiriquement : RSS stable sur 100+ épisodes réels).

        Args:
            n_episodes: Nombre d'épisodes complets à produire.
            max_attempts_multiplier: Garde-fou anti-boucle-infinie : le
                nombre total de tentatives (complètes ou non) est plafonné
                à `n_episodes * max_attempts_multiplier`. Avec le critère
                de fin corrigé (`is_finished()`, voir `simulation_engine.py`),
                le taux de complétion observé est proche de 100 %, donc ce
                plafond n'est en pratique quasiment jamais atteint.

        Raises:
            RuntimeError: si le plafond de tentatives est atteint sans
                avoir produit `n_episodes` épisodes complets — pour ne
                jamais livrer silencieusement un dataset plus petit que
                demandé.
        """
        max_attempts = max(n_episodes * max_attempts_multiplier, n_episodes + 10)
        completed_count = 0
        attempt = 0
        while completed_count < n_episodes and attempt < max_attempts:
            rows = list(self.run_episode(attempt))
            attempt += 1
            if not rows or not rows[-1]["completed"]:
                continue
            for row in rows:
                row["episode_id"] = completed_count
            yield from rows
            completed_count += 1

        if completed_count < n_episodes:
            raise RuntimeError(
                f"Seulement {completed_count}/{n_episodes} épisodes complets obtenus "
                f"après {attempt} tentatives. Dataset non généré pour éviter de livrer "
                f"un fichier plus petit que demandé silencieusement."
            )

    # Colonnes obligatoires du dataset exploration (schéma demandé), en
    # plus des colonnes de `SimulationRecord` déjà présentes dans toutes
    # les configurations.
    REQUIRED_EXPLORATION_COLUMNS = (
        "timestamp", "T_bain", "power_pct", "P_effective", "m_alu",
        "m_dross", "energy_kwh", "cost_usd", "yield_pct",
        "episode_id", "completed",
    )

    def export_csv(self, output_path: str | Path, n_episodes: int) -> Path:
        """Génère `n_episodes` épisodes complets et les exporte vers
        `output_path`.

        Malgré son nom conservé pour la rétrocompatibilité, cette méthode
        utilise désormais `DataExporter` et respecte `self.export_format`
        ("csv" ou "sqlite") plutôt que d'écrire directement du CSV.

        VALIDATION AUTOMATIQUE (audit qualité dataset) : avant d'écrire
        chaque ligne, une vérification légère est effectuée en continu
        (schéma stable d'une ligne à l'autre, colonnes obligatoires
        présentes en mode exploration) ; à la fin de la génération, le
        nombre d'épisodes distincts réellement écrits est comparé à
        `n_episodes`. Cette validation reste volontairement simple
        (comparaisons d'ensembles de clés, compteurs) : elle ne bufferise
        jamais le dataset complet en mémoire, seulement l'épisode en cours
        (déjà borné par `max_steps_per_episode`).
        """
        output_path = Path(output_path)
        row_count = 0
        expected_fieldnames: set[str] | None = None
        episode_ids_seen: set = set()

        exporter = DataExporter(output_path, export_format=self.export_format, batch_size=1000)
        try:
            for row in self.generate(n_episodes):
                row_keys = set(row.keys())
                if expected_fieldnames is None:
                    expected_fieldnames = row_keys
                    if self.exploration:
                        missing = set(self.REQUIRED_EXPLORATION_COLUMNS) - expected_fieldnames
                        if missing:
                            raise ValueError(
                                f"Colonnes obligatoires manquantes dans le dataset : {sorted(missing)}"
                            )
                elif row_keys != expected_fieldnames:
                    raise ValueError(
                        "Schéma instable détecté en cours de génération : "
                        f"colonnes attendues {sorted(expected_fieldnames)}, "
                        f"reçues {sorted(row_keys)}."
                    )
                episode_ids_seen.add(row["episode_id"])
                exporter.write_row(row)
                row_count += 1
        finally:
            exporter.close()

        if len(episode_ids_seen) != n_episodes:
            raise RuntimeError(
                f"Validation finale échouée : {len(episode_ids_seen)} épisodes distincts "
                f"écrits, {n_episodes} demandés."
            )

        print(f"Dataset généré ({self.export_format}) : {row_count} lignes, "
              f"{len(episode_ids_seen)} épisodes complets -> {output_path}")
        return output_path
