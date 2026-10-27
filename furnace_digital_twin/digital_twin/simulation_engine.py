"""
Moteur de simulation
======================

Orchestre l'ensemble des modules (matériaux, thermique, électromagnétique,
électronique de puissance, refroidissement, capteurs, vieillissement,
défauts) et fait avancer le jumeau numérique pas de temps par pas de temps.

À chaque itération :
1. lecture des paramètres / consignes (process control simplifié) ;
2. calcul physique (EM -> électronique de puissance -> thermique -> refroidissement) ;
3. mise à jour des états internes (vieillissement, défauts) ;
4. calcul des résidus (écarts modèle nominal / modèle avec défauts) ;
5. calcul des performances (rendement, énergie, pertes) ;
6. génération de l'enregistrement de télémétrie (capteurs bruités).
"""

from __future__ import annotations

import random
import warnings
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .materials import AluminumAlloy, get_alloy
from .thermal import ThermalModel
from .electromagnetic import InductionCoilModel
from .power_electronics import PowerElectronicsModel
from .cooling import CoolingSystem
from .sensors import make_standard_sensor_suite, SensorFaultMode
from .aging import AgingModel
from .faults import FaultInjector, FaultType
from .control_expert import OptimalController, ControlAction
from .safety_wrapper import safe_power_setpoint, safe_cooling_flow


class ProcessPhase(Enum):
    HEATING = "heating"
    MELTING = "melting"
    HOLDING = "holding"
    COOLING = "cooling"


@dataclass
class FurnaceConfig:
    alloy_name: str = "AL1050"
    mass_kg: float = 500.0
    initial_temp_c: float = 25.0
    ambient_temp_c: float = 25.0
    target_temp_c: float = 720.0
    hold_duration_s: float = 600.0
    max_power_w: float = 500_000.0
    frequency_hz: float = 1000.0
    grid_voltage_v: float = 400.0
    exposed_surface_m2: float = 0.6
    cooling_nominal_flow_lpm: float = 40.0
    cooling_inlet_temp_c: float = 18.0
    random_fault_probability_per_hour: float = 0.05
    seed: Optional[int] = None
    initial_wear_level: float = 0.0  # 0 = neuf, 1 = très usé
    # Si vrai, la consigne de puissance et la fraction de débit de
    # refroidissement ne sont plus calculées par la logique historique
    # (`_power_setpoint_w` + fraction nominale) mais par un contrôleur
    # prédictif (`OptimalController`). Désactivé par défaut : le
    # comportement existant est inchangé si cette option n'est pas activée.
    use_optimal_control: bool = False


@dataclass
class SimulationRecord:
    time_s: float
    phase: str
    temperature_metal_c: float
    temperature_metal_measured_c: float
    temperature_surface_measured_c: float
    melt_fraction: float
    power_setpoint_w: float
    active_power_w: float
    active_power_measured_w: float
    reactive_power_var: float
    power_factor: float
    frequency_hz: float
    frequency_measured_hz: float
    current_a: float
    current_measured_a: float
    voltage_v: float
    voltage_measured_v: float
    impedance_ohm: float
    igbt_temp_c: float
    igbt_temp_measured_c: float
    refractory_temp_c: float
    refractory_temp_measured_c: float
    water_inlet_temp_c: float
    water_outlet_temp_c: float
    water_outlet_temp_measured_c: float
    water_flow_lpm: float
    water_flow_measured_lpm: float
    water_pressure_bar: float
    water_pressure_measured_bar: float
    efficiency: float
    energy_consumed_j: float
    thermal_losses_w: float
    residual_power_w: float
    health_refractory: float
    health_coil: float
    health_capacitors: float
    health_igbt: float
    active_faults: str
    anomaly: int
    maintenance_alert: int
    alarm: str
    # Sortie brute du contrôleur optimal (avant application des garde-fous
    # de sécurité). NaN si `use_optimal_control` est désactivé, pour que
    # ces colonnes restent présentes et exploitables dans le dataset même
    # lorsque l'option n'est pas utilisée sur un épisode donné.
    optimal_power_setpoint_w: float
    optimal_cooling_flow_fraction: float
    # --- Puissance variable (5-100 %), coûts et rendement matière ---
    # `power_pct` : fraction de puissance nominale effectivement appliquée
    # ce pas-ci (NaN si le pas n'a pas été piloté via le paramètre
    # `power_pct` de `step()`, c.-à-d. logique historique/contrôleur).
    power_pct: float
    # `p_effective_w` : consigne de puissance réellement appliquée après
    # tous les garde-fous (dont la coupure de sécurité thermique absolue),
    # équivalente à `power_setpoint_w` mais nommée explicitement pour
    # correspondre à P_effective = P_nominale * power_pct.
    p_effective_w: float
    m_dross_kg: float          # masse cumulée de calamine/dross (kg)
    energy_kwh: float          # énergie cumulée consommée (kWh)
    cost_usd: float            # coût cumulé (énergie + pertes matière)
    yield_pct: float           # rendement matière cumulé (%)


class FurnaceDigitalTwin:
    """Jumeau numérique complet d'un four à induction pour le recyclage
    de l'aluminium."""

    # Sécurité thermique absolue : quelle que soit la source de la
    # consigne de puissance (logique historique, contrôleur optimal, ou
    # `power_pct` explicite passé à `step()`), l'injection de puissance
    # est coupée dès que le bain atteint ou dépasse cette température.
    # Volontairement une constante fixe (indépendante de
    # `config.target_temp_c`, que l'utilisateur pourrait régler plus
    # haut) : c'est un garde-fou de sécurité, pas un paramètre de procédé.
    ABSOLUTE_MAX_BATH_TEMP_C: float = 720.0

    # Bornes admissibles pour le paramètre `power_pct` de `step()`.
    MIN_POWER_PCT: float = 0.05
    MAX_POWER_PCT: float = 1.00

    # Coefficient empirique de formation de calamine/dross (kg par m^2 de
    # surface libre et par seconde, à pleine puissance et bain totalement
    # liquide). Calibré pour donner des pertes de l'ordre de 1 à 5 % de la
    # charge sur un cycle complet, plage usuelle en recyclage d'aluminium.
    #
    # CORRECTION (audit qualité IA) : la valeur précédente (4e-6) ne
    # produisait, empiriquement, qu'une perte de l'ordre de 0,0006 % de la
    # charge (`yield_pct` quasi constant à 99,9998 % sur tout dataset
    # généré) — trois à quatre ordres de grandeur en dessous de l'objectif
    # documenté ci-dessus. `yield_pct`/`cost_usd` (qui en dérive) étaient de
    # ce fait inexploitables comme cible de régression (variance quasi
    # nulle). Recalibré à 1e-2 : validé empiriquement sur un batch
    # d'épisodes exploration variés (masse/alliage/puissance) -> `yield_pct`
    # couvre désormais ~94,6-99,5 % (std ≈ 0,96), conforme à la fourchette
    # 1-5 % documentée. Seule cette constante change ; aucune autre
    # formule du modèle de dross n'est modifiée.
    DROSS_RATE_COEFFICIENT_KG_M2_S: float = 1e-2

    # Tarifs utilisés pour le calcul du coût cumulé (USD).
    ENERGY_COST_USD_PER_KWH: float = 0.25
    DROSS_COST_USD_PER_KG: float = 0.60

    # Fraction de `config.max_power_w` que l'on vise à rendre réellement
    # délivrable (limitée par le bus DC) lors du dimensionnement
    # automatique du couplage électromagnétique ci-dessous (voir
    # `__init__`). 0.7 laisse une marge réaliste (le circuit n'est jamais
    # taillé "pile", comme sur une installation réelle) tout en évitant
    # le plafond fixe ~26 kW observé quelle que soit la consigne.
    _TARGET_ACHIEVABLE_POWER_FRACTION: float = 0.7
    _MIN_MATCHING_TRANSFORMER_RATIO: float = 3.0
    _MAX_MATCHING_TRANSFORMER_RATIO: float = 60.0

    def __init__(self, config: FurnaceConfig) -> None:
        if config.mass_kg <= 0:
            raise ValueError("mass_kg doit être strictement positif")
        self.config = config
        self.rng = random.Random(config.seed)

        self.alloy: AluminumAlloy = get_alloy(config.alloy_name)
        self.thermal = ThermalModel(
            alloy=self.alloy,
            mass_kg=config.mass_kg,
            exposed_surface_m2=config.exposed_surface_m2,
            ambient_temp_c=config.ambient_temp_c,
            initial_temp_c=config.initial_temp_c,
        )
        # BUG CORRIGÉ (audit dataset — épisodes jamais terminés) :
        # `InductionCoilModel()` était toujours construit avec sa
        # géométrie par défaut, totalement indépendante de
        # `config.max_power_w`. Or la puissance active réellement
        # délivrable est plafonnée par le bus DC (~V_bus² / R_tank) et ne
        # dépend donc PAS de la consigne demandée : quelle que soit
        # `max_power_w` (150 kW à 900 kW en génération standard), la
        # puissance livrée plafonnait à ~26 kW, rendant le chauffage de
        # charges de plusieurs centaines de kg impossible dans le budget
        # de pas de temps -> `completed=False` sur la quasi-totalité des
        # épisodes générés. Correction minimale : on dimensionne
        # `matching_transformer_ratio` (paramètre déjà prévu à cet effet
        # dans `InductionCoilModel`) pour que la puissance délivrable au
        # point de fonctionnement initial représente une fraction réaliste
        # de `max_power_w`, sans toucher au modèle électromagnétique.
        _probe_em = InductionCoilModel()
        _probe_state = _probe_em.compute_state(
            self.alloy, config.initial_temp_c, config.frequency_hz, melt_fraction=0.0
        )
        _bus_voltage_v = 1.35 * config.grid_voltage_v * 0.98  # cf. PowerElectronicsModel
        _r_secondary_ohm = (
            _probe_state.total_resistance_ohm / (_probe_em.matching_transformer_ratio ** 2)
        )
        _target_achievable_w = self._TARGET_ACHIEVABLE_POWER_FRACTION * config.max_power_w
        if _target_achievable_w > 0 and _r_secondary_ohm > 0:
            _scaled_ratio = _bus_voltage_v / math.sqrt(_target_achievable_w * _r_secondary_ohm)
            _scaled_ratio = max(
                self._MIN_MATCHING_TRANSFORMER_RATIO,
                min(self._MAX_MATCHING_TRANSFORMER_RATIO, _scaled_ratio),
            )
        else:
            _scaled_ratio = _probe_em.matching_transformer_ratio
        self.em_model = InductionCoilModel(matching_transformer_ratio=_scaled_ratio)
        self.pe_model = PowerElectronicsModel(
            grid_voltage_v=config.grid_voltage_v,
            ambient_temp_c=config.ambient_temp_c,
        )
        self.cooling = CoolingSystem(
            nominal_flow_rate_lpm=config.cooling_nominal_flow_lpm,
            inlet_temp_c=config.cooling_inlet_temp_c,
        )
        self.sensors = make_standard_sensor_suite()
        self.aging = AgingModel()
        # État initial de vieillissement (permet de simuler un four déjà usé)
        wear = max(0.0, min(1.0, config.initial_wear_level))
        # CORRECTION (audit qualité IA) : avec le multiplicateur précédent
        # (0.6), le plancher de `health_refractory` à usure initiale
        # maximale (wear=1.0) était de 0.4 — *au-dessus* du seuil
        # `min_health < 0.3` qui déclenche `maintenance_alert`
        # (`simulation_engine.py`, cf. plus bas). Ce seuil était donc
        # structurellement inatteignable, quelle que soit la distribution
        # d'usure tirée par le générateur : `maintenance_alert` valait 0
        # sur 100 % des lignes de tout dataset généré (constaté sur le
        # dataset audité). Un premier relevage à 0.85 rendait le seuil
        # techniquement atteignable mais seulement pour ~0,2 % des tirages
        # de `betavariate(1.5, 4.0)` (0 déclenchement observé sur 200
        # épisodes de test) — trop rare pour être une classe positive
        # exploitable à taille de dataset raisonnable (1000-3000
        # épisodes). Relevé à 0.95 (plancher 0.05) : ~1 % des tirages
        # dépassent alors le seuil d'usure initiale nécessaire (validé par
        # calcul direct sur la distribution), soit ~10 épisodes/1000 rien
        # que par l'usure initiale, plus les cas limites qui franchissent
        # le seuil en cours d'épisode via la dégradation in-situ
        # (`AgingModel.step`). Seul ce multiplicateur change, les trois
        # autres composants (coil/capacitors/igbt) restent inchangés.
        self.aging.health.refractory = 1.0 - wear * 0.95
        self.aging.health.coil = 1.0 - wear * 0.5
        self.aging.health.capacitors = 1.0 - wear * 0.5
        self.aging.health.igbt = 1.0 - wear * 0.4

        self.fault_injector = FaultInjector(
            random_fault_probability_per_hour=config.random_fault_probability_per_hour,
            rng=self.rng,
            sensor_names=list(self.sensors.keys()),
        )

        self.refractory_temp_c = config.ambient_temp_c
        self.time_s = 0.0
        self.energy_consumed_j = 0.0
        self.phase = ProcessPhase.HEATING
        self._hold_elapsed_s = 0.0
        self.nominal_coil_current_a = 800.0
        # Masse cumulée de calamine/dross formée depuis le début du cycle
        # (kg). Ne modifie volontairement pas `self.thermal.mass_kg` (la
        # perte reste marginale devant la masse totale sur un cycle
        # normal) afin de ne pas complexifier le modèle thermique
        # existant ; c'est un compteur de bilan matière/coût en parallèle.
        self.m_dross_kg = 0.0

        # --- Contrôle optimal (optionnel) ---
        self.controller: Optional[OptimalController] = None
        if config.use_optimal_control:
            self.controller = OptimalController(
                max_power_w=config.max_power_w,
                target_temp_c=config.target_temp_c,
            )
        # États mémorisés d'un pas à l'autre, utilisés par le contrôleur
        # (gradient de température) et par les garde-fous de sécurité
        # (limitation de rampe par rapport à la dernière consigne appliquée).
        self._previous_temp_c = config.initial_temp_c
        # Puissance active *physiquement délivrée* au pas précédent (issue
        # de `pe_state.active_power_w`, donc potentiellement limitée par le
        # circuit/bus DC bien en-deçà de la consigne) : sert d'entrée au
        # contrôleur pour l'estimation du gain thermique.
        self._previous_power_w = 0.0
        # Dernière *consigne* de puissance effectivement appliquée (après
        # garde-fous), utilisée comme référence de rampe par
        # `safe_power_setpoint`. Distincte de `_previous_power_w` : si la
        # consigne dépasse ce que le circuit peut délivrer, la rampe doit
        # continuer à progresser à partir de la consigne précédente, pas
        # à partir de la puissance physiquement atteinte (sous peine de
        # ré-déclencher un écrêtage de rampe à chaque pas).
        self._previous_power_setpoint_applied_w = 0.0
        self._previous_igbt_temp_c = config.ambient_temp_c
        self._previous_cooling_flow_fraction = 1.0
        # Gradient de température (°C/s) mesuré à l'issue du pas précédent.
        # Calculé une fois pour toutes en fin de `step()` (une fois la
        # température réellement mise à jour) puis réutilisé en tout début
        # du pas suivant pour alimenter le contrôleur : le recalculer en
        # tout début de `step()` donnerait toujours 0, car à ce moment-là
        # `self.thermal.state.temperature_c` n'a pas encore été mis à jour
        # pour le pas courant et vaut donc encore `self._previous_temp_c`.
        self._last_dtemp_dt_c_per_s = 0.0

    def _select_phase(self) -> ProcessPhase:
        temp = self.thermal.state.temperature_c
        if self.phase == ProcessPhase.HEATING:
            if temp >= self.alloy.solidus_c:
                return ProcessPhase.MELTING
            return ProcessPhase.HEATING
        if self.phase == ProcessPhase.MELTING:
            if temp >= self.config.target_temp_c and self.thermal.state.melt_fraction >= 1.0:
                return ProcessPhase.HOLDING
            return ProcessPhase.MELTING
        if self.phase == ProcessPhase.HOLDING:
            if self._hold_elapsed_s >= self.config.hold_duration_s:
                return ProcessPhase.COOLING
            return ProcessPhase.HOLDING
        return ProcessPhase.COOLING

    def _power_setpoint_w(self, phase: ProcessPhase) -> float:
        if phase in (ProcessPhase.HEATING, ProcessPhase.MELTING):
            return self.config.max_power_w
        if phase == ProcessPhase.HOLDING:
            return self.config.max_power_w * 0.25
        return 0.0

    def step(self, dt_s: float, power_pct: Optional[float] = None) -> SimulationRecord:
        """Fait avancer la simulation d'un pas de temps.

        Args:
            dt_s: Durée du pas de temps (s).
            power_pct: Si fourni, fraction de puissance nominale
                (0.05 à 1.00, soit 5 % à 100 %) à appliquer ce pas-ci :
                P_effective = `config.max_power_w` * power_pct. Cette
                consigne explicite prend le pas sur la logique historique
                de phase et sur le contrôleur optimal éventuel (elle les
                *remplace* pour ce pas, elle ne s'y ajoute pas). Une
                valeur hors de [0.05, 1.00] est ramenée dans l'intervalle
                et un avertissement est émis. Si `None` (valeur par
                défaut), le comportement est strictement inchangé par
                rapport aux versions précédentes (rétrocompatibilité).

                Dans tous les cas — `power_pct` fourni ou non — la
                sécurité thermique absolue reste active : dès que le bain
                atteint `ABSOLUTE_MAX_BATH_TEMP_C` (720 °C), l'injection
                de puissance est coupée quelle que soit la consigne
                demandée.
        """
        t = self.time_s
        self.phase = self._select_phase()
        if self.phase == ProcessPhase.HOLDING:
            self._hold_elapsed_s += dt_s

        active_faults = self.fault_injector.step(t, dt_s)

        # --- Défauts affectant les grandeurs d'entrée ---
        grid_sag = self.fault_injector.get_severity(FaultType.GRID_VOLTAGE_SAG)
        grid_voltage = self.config.grid_voltage_v * (1.0 - 0.4 * grid_sag)

        cooling_loss = self.fault_injector.get_severity(FaultType.COOLING_LOSS)
        fault_flow_fraction = max(0.0, 1.0 - cooling_loss)

        capacitor_fault = self.fault_injector.get_severity(FaultType.CAPACITOR_DEGRADED)
        capacitor_health_effective = self.aging.health.capacitors * (1.0 - 0.7 * capacitor_fault)

        coil_fault = self.fault_injector.get_severity(FaultType.COIL_DAMAGED)
        igbt_overheat_fault = self.fault_injector.get_severity(FaultType.IGBT_OVERHEAT)
        short_circuit_fault = self.fault_injector.get_severity(FaultType.PARTIAL_SHORT_CIRCUIT)
        impurities_fault = self.fault_injector.get_severity(FaultType.METAL_IMPURITIES)

        # --- Électromagnétique ---
        em_state = self.em_model.compute_state(
            self.alloy,
            self.thermal.state.temperature_c,
            self.config.frequency_hz,
            melt_fraction=self.thermal.state.melt_fraction,
        )
        if coil_fault > 0:
            em_state.coil_resistance_ohm *= (1.0 + 2.0 * coil_fault)
            em_state.total_resistance_ohm = em_state.coil_resistance_ohm + em_state.load_resistance_ohm
        if short_circuit_fault > 0:
            em_state.total_resistance_ohm *= max(0.05, 1.0 - 0.8 * short_circuit_fault)
        if impurities_fault > 0:
            # Les impuretés modifient la résistivité effective de la charge
            em_state.load_resistance_ohm *= (1.0 + 0.5 * impurities_fault)
            em_state.total_resistance_ohm = em_state.coil_resistance_ohm + em_state.load_resistance_ohm

        # --- Consigne de puissance / refroidissement ---
        # Logique historique par défaut ; surchargée par le contrôleur
        # optimal si `use_optimal_control` est activé dans la config.
        # BUG CORRIGÉ (audit dataset — colonnes 100% NaN) : ces deux
        # colonnes valaient NaN par défaut quand `use_optimal_control` est
        # désactivé (le cas par défaut), ce qui produisait des colonnes
        # entièrement NaN dans tout dataset généré sans
        # `--with-optimal-control`. 0.0 est un sentinel neutre et
        # documenté ("contrôleur optimal non utilisé pour ce pas") plutôt
        # qu'une valeur non exploitable par une IA.
        optimal_power_setpoint_w = 0.0
        optimal_cooling_flow_fraction = 0.0

        if self.controller is not None:
            phase_allows_heating = self.phase in (
                ProcessPhase.HEATING, ProcessPhase.MELTING, ProcessPhase.HOLDING,
            )
            action: ControlAction = self.controller.compute_action(
                current_temp_c=self.thermal.state.temperature_c,
                dtemp_dt_c_per_s=self._last_dtemp_dt_c_per_s,
                current_power_w=self._previous_power_w,
                igbt_temp_c=self._previous_igbt_temp_c,
                phase_allows_heating=phase_allows_heating,
                dt_s=dt_s,
            )
            optimal_power_setpoint_w = action.power_setpoint_w
            optimal_cooling_flow_fraction = action.cooling_flow_fraction

            # Garde-fous de sécurité indépendants du contrôleur : bornent la
            # consigne et sa rampe quelle que soit son origine.
            power_setpoint = safe_power_setpoint(
                desired_power_w=action.power_setpoint_w,
                previous_power_w=self._previous_power_setpoint_applied_w,
                dt_s=dt_s,
                max_power_w=self.config.max_power_w,
            )
            controlled_flow_fraction = safe_cooling_flow(
                desired_flow_fraction=action.cooling_flow_fraction,
                previous_flow_fraction=self._previous_cooling_flow_fraction,
                dt_s=dt_s,
            )
            # Un défaut de refroidissement physique reste prioritaire sur la
            # volonté du contrôleur (on ne peut pas demander plus de débit
            # que ce que la pompe/le circuit peuvent réellement fournir).
            flow_fraction = min(controlled_flow_fraction, fault_flow_fraction)
        else:
            power_setpoint = self._power_setpoint_w(self.phase)
            flow_fraction = fault_flow_fraction

        # --- Puissance variable explicite (5-100 %) ---
        # Remplace la consigne issue de la logique historique/du
        # contrôleur pour ce pas-ci si `power_pct` est fourni.
        explicit_power_pct_applied: Optional[float] = None
        if power_pct is not None:
            clamped_power_pct = min(self.MAX_POWER_PCT, max(self.MIN_POWER_PCT, power_pct))
            if clamped_power_pct != power_pct:
                warnings.warn(
                    f"power_pct={power_pct:.4f} hors bornes "
                    f"[{self.MIN_POWER_PCT}, {self.MAX_POWER_PCT}] -> ramené à "
                    f"{clamped_power_pct:.4f}.",
                    stacklevel=2,
                )
            explicit_power_pct_applied = clamped_power_pct
            power_setpoint = self.config.max_power_w * clamped_power_pct

        # --- Sécurité thermique absolue ---
        # Coupe l'injection de puissance dès que le bain atteint le seuil
        # critique, quelle que soit l'origine de la consigne ci-dessus.
        if self.thermal.state.temperature_c >= self.ABSOLUTE_MAX_BATH_TEMP_C:
            power_setpoint = 0.0

        # BUG CORRIGÉ (audit dataset — colonne `power_pct` 100% NaN) :
        # `power_pct_applied` ne valait une fraction que si `power_pct`
        # était explicitement passé à `step()` (mode exploration), et
        # restait NaN sinon (mode standard, ou pas coupé par la sécurité
        # thermique) -> colonne entièrement NaN dans tout dataset généré
        # sans passer `power_pct`. Elle reflète maintenant, dans tous les
        # cas, la fraction de `max_power_w` réellement appliquée après
        # tous les garde-fous (y compris la coupure de sécurité absolue).
        if explicit_power_pct_applied is not None and power_setpoint > 0:
            power_pct_applied = explicit_power_pct_applied
        else:
            power_pct_applied = (
                power_setpoint / self.config.max_power_w if self.config.max_power_w > 0 else 0.0
            )

        pe_state = self.pe_model.compute_state(
            target_power_w=power_setpoint,
            em_state=em_state,
            grid_voltage_v=grid_voltage,
            capacitor_health=max(0.05, capacitor_health_effective),
            igbt_health=max(0.05, self.aging.health.igbt),
        )
        igbt_temp = pe_state.igbt_junction_temp_c
        if igbt_overheat_fault > 0:
            igbt_temp += 60.0 * igbt_overheat_fault

        # --- Refroidissement ---
        heat_to_remove_w = pe_state.igbt_losses_w + em_state.coil_resistance_ohm * (
            pe_state.inverter_output_current_a ** 2
        )
        cooling_state = self.cooling.step(
            heat_to_remove_w=heat_to_remove_w,
            flow_fraction=flow_fraction,
            inlet_temp_c=self.config.cooling_inlet_temp_c,
        )

        # --- Thermique (charge métallique) ---
        # Le réfractaire tend lentement vers la température du métal (inertie)
        self.refractory_temp_c += 0.002 * dt_s * (
            self.thermal.state.temperature_c * 0.6 - self.refractory_temp_c
        )
        thermal_state = self.thermal.step(
            power_in_w=pe_state.active_power_w * em_state.coupling_efficiency,
            dt_s=dt_s,
            refractory_temp_c=self.refractory_temp_c,
        )
        # Sécurité thermique absolue (suite) : un pas de temps assez grand
        # ou une consigne encore ouverte au moment où le seuil est franchi
        # peuvent faire dépasser 720 °C en un seul pas (overshoot), avant
        # que la coupure de puissance ci-dessus ne s'applique au pas
        # suivant. On écrête donc aussi la température obtenue, sans
        # ajouter de contrôleur prédictif : simple clamp après coup.
        if thermal_state.temperature_c > self.ABSOLUTE_MAX_BATH_TEMP_C:
            thermal_state.temperature_c = self.ABSOLUTE_MAX_BATH_TEMP_C
            self.thermal.state.temperature_c = self.ABSOLUTE_MAX_BATH_TEMP_C
        losses = self.thermal.losses_w(self.refractory_temp_c)

        # --- Formation de calamine (dross) ---
        # Modèle empirique simplifié : le taux de formation croît avec la
        # fraction fondue de la charge (surface libre oxydable exposée à
        # l'air) et avec l'intensité du brassage électromagnétique (proxy :
        # fraction de puissance nominale réellement appliquée). Le
        # cumul est plafonné à la masse initiale de la charge.
        applied_power_fraction = (
            power_setpoint / self.config.max_power_w if self.config.max_power_w > 0 else 0.0
        )
        dross_rate_kg_s = (
            self.DROSS_RATE_COEFFICIENT_KG_M2_S
            * self.config.exposed_surface_m2
            * thermal_state.melt_fraction
            * (0.4 + 0.6 * applied_power_fraction)
        )
        self.m_dross_kg = min(
            self.config.mass_kg, self.m_dross_kg + dross_rate_kg_s * dt_s
        )

        # --- Vieillissement ---
        health = self.aging.step(
            dt_s=dt_s,
            metal_temp_c=thermal_state.temperature_c,
            igbt_temp_c=igbt_temp,
            capacitor_stress_factor=pe_state.capacitor_stress_factor,
            coil_current_a=pe_state.inverter_output_current_a,
            coil_current_nominal_a=self.nominal_coil_current_a,
            is_heating=self.phase in (ProcessPhase.HEATING, ProcessPhase.MELTING),
        )

        # --- Bilan énergétique et rendement ---
        # Énergie facturée basée sur la puissance ACTIVE (celle qui est
        # réellement facturée par le fournisseur d'électricité), pas la
        # puissance apparente (qui inclut la composante réactive).
        self.energy_consumed_j += pe_state.active_power_w * dt_s
        useful_power = pe_state.active_power_w * em_state.coupling_efficiency
        efficiency = useful_power / pe_state.apparent_power_va if pe_state.apparent_power_va > 0 else 0.0
        residual_power_w = pe_state.active_power_w - useful_power - losses["total_w"]

        # --- Coûts et rendement matière (cumulés depuis le début du cycle) ---
        # m_initiale = self.config.mass_kg (masse initiale de la charge,
        # typiquement 1 à 100 kg pour les épisodes en mode exploration).
        energy_kwh = self.energy_consumed_j / 3.6e6
        m_initiale = self.config.mass_kg
        yield_pct = (
            ((m_initiale - self.m_dross_kg) / m_initiale) * 100.0 if m_initiale > 0 else 0.0
        )
        cost_usd = (energy_kwh * self.ENERGY_COST_USD_PER_KWH) + (
            self.m_dross_kg * self.DROSS_COST_USD_PER_KG
        )

        # --- Capteurs (mesures bruitées) ---
        sensor_fault_active = FaultType.SENSOR_FAILURE in self.fault_injector.active_types()
        if sensor_fault_active:
            for f in self.fault_injector.active_faults:
                if f.fault_type == FaultType.SENSOR_FAILURE and f.target_sensor:
                    self.sensors[f.target_sensor].set_fault(SensorFaultMode.OUT_OF_RANGE)
        else:
            for sensor in self.sensors.values():
                sensor.clear_fault()

        temp_measured = self.sensors["thermocouple_metal"].read(thermal_state.temperature_c, dt_s, self.rng)
        surface_measured = self.sensors["pyrometer_surface"].read(thermal_state.temperature_c * 0.95, dt_s, self.rng)
        current_measured = self.sensors["current_sensor"].read(pe_state.inverter_output_current_a, dt_s, self.rng)
        voltage_measured = self.sensors["voltage_sensor"].read(pe_state.dc_bus_voltage_v, dt_s, self.rng)
        power_measured = self.sensors["power_sensor"].read(pe_state.active_power_w, dt_s, self.rng)
        freq_measured = self.sensors["frequency_sensor"].read(self.config.frequency_hz, dt_s, self.rng)
        flow_measured = self.sensors["flow_sensor"].read(cooling_state.flow_rate_lpm, dt_s, self.rng)
        pressure_measured = self.sensors["pressure_sensor"].read(cooling_state.pressure_bar, dt_s, self.rng)
        igbt_measured = self.sensors["igbt_temp_sensor"].read(igbt_temp, dt_s, self.rng)
        refractory_measured = self.sensors["refractory_temp_sensor"].read(self.refractory_temp_c, dt_s, self.rng)
        water_out_measured = self.sensors["water_temp_sensor"].read(cooling_state.outlet_temp_c, dt_s, self.rng)

        # --- Détection d'anomalie et alerte maintenance (labels IA) ---
        anomaly = 1 if active_faults or not cooling_state.is_sufficient or igbt_temp > 150 else 0
        min_health = min(health.refractory, health.coil, health.capacitors, health.igbt)
        maintenance_alert = 1 if min_health < 0.3 else 0
        alarm_list = [f.fault_type.value for f in active_faults]
        if igbt_temp > 150:
            alarm_list.append("igbt_overtemp")
        if not cooling_state.is_sufficient:
            alarm_list.append("cooling_insufficient")
        alarm = ",".join(sorted(set(alarm_list))) if alarm_list else ""

        record = SimulationRecord(
            time_s=t,
            phase=self.phase.value,
            temperature_metal_c=thermal_state.temperature_c,
            temperature_metal_measured_c=temp_measured if temp_measured is not None else float("nan"),
            temperature_surface_measured_c=surface_measured if surface_measured is not None else float("nan"),
            melt_fraction=thermal_state.melt_fraction,
            power_setpoint_w=power_setpoint,
            active_power_w=pe_state.active_power_w,
            active_power_measured_w=power_measured if power_measured is not None else float("nan"),
            reactive_power_var=pe_state.reactive_power_var,
            power_factor=pe_state.power_factor,
            frequency_hz=self.config.frequency_hz,
            frequency_measured_hz=freq_measured if freq_measured is not None else float("nan"),
            current_a=pe_state.inverter_output_current_a,
            current_measured_a=current_measured if current_measured is not None else float("nan"),
            voltage_v=pe_state.dc_bus_voltage_v,
            voltage_measured_v=voltage_measured if voltage_measured is not None else float("nan"),
            impedance_ohm=em_state.impedance_magnitude_ohm,
            igbt_temp_c=igbt_temp,
            igbt_temp_measured_c=igbt_measured if igbt_measured is not None else float("nan"),
            refractory_temp_c=self.refractory_temp_c,
            refractory_temp_measured_c=refractory_measured if refractory_measured is not None else float("nan"),
            water_inlet_temp_c=cooling_state.inlet_temp_c,
            water_outlet_temp_c=cooling_state.outlet_temp_c,
            water_outlet_temp_measured_c=water_out_measured if water_out_measured is not None else float("nan"),
            water_flow_lpm=cooling_state.flow_rate_lpm,
            water_flow_measured_lpm=flow_measured if flow_measured is not None else float("nan"),
            water_pressure_bar=cooling_state.pressure_bar,
            water_pressure_measured_bar=pressure_measured if pressure_measured is not None else float("nan"),
            efficiency=efficiency,
            energy_consumed_j=self.energy_consumed_j,
            thermal_losses_w=losses["total_w"],
            residual_power_w=residual_power_w,
            health_refractory=health.refractory,
            health_coil=health.coil,
            health_capacitors=health.capacitors,
            health_igbt=health.igbt,
            active_faults=",".join(f.fault_type.value for f in active_faults),
            anomaly=anomaly,
            maintenance_alert=maintenance_alert,
            alarm=alarm,
            optimal_power_setpoint_w=optimal_power_setpoint_w,
            optimal_cooling_flow_fraction=optimal_cooling_flow_fraction,
            power_pct=power_pct_applied,
            p_effective_w=power_setpoint,
            m_dross_kg=self.m_dross_kg,
            energy_kwh=energy_kwh,
            cost_usd=cost_usd,
            yield_pct=yield_pct,
        )

        # Mémorisation pour le pas de temps suivant (contrôleur + rampes)
        self._last_dtemp_dt_c_per_s = (
            (thermal_state.temperature_c - self._previous_temp_c) / dt_s if dt_s > 0 else 0.0
        )
        self._previous_temp_c = thermal_state.temperature_c
        self._previous_power_w = pe_state.active_power_w
        self._previous_power_setpoint_applied_w = power_setpoint
        self._previous_igbt_temp_c = igbt_temp
        self._previous_cooling_flow_fraction = flow_fraction

        self.time_s += dt_s
        return record

    def is_finished(self) -> bool:
        """Un épisode (un « heat ») est terminé dès que le bain a atteint
        sa consigne de maintien puis a été coulé (entrée en phase
        `COOLING`).

        BUG CORRIGÉ (audit qualité dataset) : le critère précédent exigeait
        en plus que le bain refroidisse *passivement* (rayonnement +
        convection naturelle, sans refroidissement forcé du métal lui-même)
        jusqu'à `ambient_temp_c + 5°C` avant de considérer l'épisode
        terminé. Pour des charges de plusieurs dizaines de kg, ce
        refroidissement passif dans l'air prend plusieurs heures — bien
        au-delà du budget de pas de temps d'un épisode — et bloquait donc
        la majorité des épisodes en phase `cooling` sans jamais les
        terminer (`completed=False` systématique), alors que rien
        d'anormal ne s'était produit. Dans un four industriel réel, le
        cycle se termine à la coulée (le métal liquide est transvasé et
        refroidit hors du four) : la fin de l'épisode est donc désormais
        l'entrée en phase `COOLING` elle-même, pas le retour du bain à la
        température ambiante.
        """
        return self.phase == ProcessPhase.COOLING
