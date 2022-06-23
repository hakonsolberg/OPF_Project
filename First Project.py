"""
Created by Håkon Sølberg at SINTEF Energy, June 2022 - August 2022 based on a three-bus system on page 126 in the book "Power System Analysis" by P.S.R Murty
Goal: Develop a method to test OPF in real-time for the distribution network, mainly comparing convergence and computational time.
Test system: Simple 3-bus system with generation at node 1 & 2, with loads at bus 2 & 3.

Some parts of this code is based on an example for a DCOPF-system in the course "Power Markets" at NTNU.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

Ref_node=1;
Vbase=115
Sbase=100

#Y_bus=np.array([[]])

def Read_Excel(name):

    """
    Reads input file. Separate between each sheet, and store in one dictionary.
    """

    data = {}                                                                                                               #Dictionary storing the input data

    Excel_sheets = ["Node data", "Line parameters", "Demand variability"]                                                   #Name for each Excel-sheet
    Data_names = {"Node data":"Nodes", "Line parameters": "Lines", "Demand variability": "Time"}                            #Names for each sheet
    Num_names = {"Node data": "NumNodes", "Line parameters": "NumLines"}                                                    #Names for numbering
    List_names = {"Node data":"List_of_nodes", "Line parameters": "List_of_lines","Demand variability": "List_of_times"}    #Names for numbering

    for sheet in Excel_sheets:                                                                                              #For each sheet
        df = pd.read_excel(name, sheet_name = sheet, skiprows = 1)                                                          #Read sheet and excludes title.
        df = df.set_index(df_columns[0])                                                                                    #Set first column as index
        num=len(df.loc[:])                                                                                                  #Length of dataframe
        df=df.to_dict()

        df[Num_Names[sheet]] = num                                                                                          #Store length of dataframe in dictionary
        df[List_names[sheet]] = np.arange(1,num+1)

        data[Data_names[sheet]] = df                                                                                        #Store dataframe in dictionary

        data["Reference node"] = Ref_node                                                                                   #Reference node where delta = 0

    return(data)


    def OPF_model(Data):

        """
        -- Setting up the optimization model, run it and display the results. ---
        """

        model = pyo.ConcreteModel

        """
        ***** Sets (lists with nodes and lines) ****
        """

        model.Lines = pyo.Set(ordered = True, initialize = Data["Lines"]["List_of_lines"])                                  #Set of AC lines
        model.Nodes = pyo.Set(ordered = True, initialize = Data["Nodes"]["List_of_nodes"])                                  #Set for nodes
        model.GenUnits = pyo.Set(ordered = True, initialize = [1,2])                                                        #Set of generator units (1 in node 1 & 2 in node 2)
        model.LoadUnits = pyo.Set(ordered = True, initialize = [1,2])

