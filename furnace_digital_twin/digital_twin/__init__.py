"""
Jumeau numérique industriel d'un four à induction pour le recyclage de
l'aluminium.

Sous-modules :
    materials            - propriétés des alliages d'aluminium
    thermal              - modèle thermique de la charge
    electromagnetic      - modèle du couplage bobine/charge
    power_electronics    - redresseur / bus DC / onduleur IGBT
    cooling              - circuit de refroidissement
    sensors              - capteurs virtuels bruités
    aging                - vieillissement des composants
    faults               - injection de défauts
    control_expert       - contrôleur prédictif optionnel (MPC simplifié)
    safety_wrapper       - garde-fous de sécurité (bornes/rampes) sur les consignes
    data_exporter        - export CSV/SQLite par lots
    simulation_engine    - moteur de simulation (orchestrateur)
    database_generator   - génération du dataset multi-épisodes
"""

from .simulation_engine import FurnaceDigitalTwin, FurnaceConfig, SimulationRecord
from .database_generator import DatasetGenerator, ParameterSampler
from .control_expert import OptimalController, ControlAction
from .safety_wrapper import safe_power_setpoint, safe_cooling_flow
from .data_exporter import DataExporter

__all__ = [
    "FurnaceDigitalTwin",
    "FurnaceConfig",
    "SimulationRecord",
    "DatasetGenerator",
    "ParameterSampler",
    "OptimalController",
    "ControlAction",
    "safe_power_setpoint",
    "safe_cooling_flow",
    "DataExporter",
]

__version__ = "0.2.0"
