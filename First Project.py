"""
Created by Håkon Sølberg at SINTEF Energy, June 2022 - August 2022.
Goal: Develop a method to test OPF in real-time for the distribution network.
Test system: Simple 3-bus system with generation at node 1 & 2, with loads at bus 2 & 3.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

Vbase=115
Sbase=100