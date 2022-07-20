"""
Created by Håkon Sølberg at SINTEF Energy, June 2022 - August 2022 based on a three-bus system on page 126 in the book "Power System Analysis" by P.S.R Murty
Goal: Develop a method to test OPF in real-time for the distribution network, mainly comparing convergence and computational time.
Test system: Simple 3-bus system with generation at node 1 & 2, with loads at bus 2 & 3.

Some parts of this code is based on an example for a DCOPF-system in the course "Power Markets" at NTNU.
The
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

Ref_node=1;
Vbase=115
Sbase=100

#Y_bus=np.array([[6.25-18.75*j -1.25+3.75*j -5+15*j],
#                [-1.25+3.73*j 2.9167-8.75*j -1.6667+5*j],
#                [-5+15*j -1.6667+5*j 6.6667 -20*j]])

Y_Bus = np.array([[19.764, 3.95, 15.81],                                                                                         #Defining the magnitude of the admittance bus
                 [3.93, 9.22, 5.27],
                 [15.81, 5.27, 21.08]])

#Theta = np.array([-71.57, 108.43, 108.44],                                                                                      #Angles of the admittance
#                 [108.53, -71.56, 108.44],
#                 [108.44, 108.44, -71.57])

Theta = np.array([[-1.249, 1.892, 0.276],
                 [1.894, -1.249, 1.894],
                 [1.894, 1.894, -1.249]])
def Main():

    "Main function that set up the problem and execute the code"

    global Data
    Data = Read_Excel("Data_3bus.xlsx")

    OPF_model(Data)                                                                                                         #Run the model with the set data

    return()

def Read_Excel(name):

    """
    Reads input file. Separate between each sheet, and store in one dictionary.
    """

    data = {}                                                                                                               #Dictionary storing the input data

    Excel_sheets = ["Node data", "Line parameters", "Demand variability"]                                                   #Name for each Excel-sheet
    Data_names = {"Node data":"Nodes", "Line parameters": "Lines", "Demand variability": "Time"}                            #Names for each sheet
    Num_names = {"Node data": "NumNodes", "Line parameters": "NumLines", "Demand variability": "NumTimes"}                  #Names for numbering
    List_names = {"Node data":"List_of_nodes", "Line parameters": "List_of_lines","Demand variability": "List_of_times"}    #Names for numbering

    for sheet in Excel_sheets:                                                                                              #For each sheet
        df = pd.read_excel(name, sheet_name = sheet, skiprows = 1)                                                          #Read sheet and excludes title.
        df = df.set_index(df.columns[0])                                                                                    #Set first column as index
        num=len(df.loc[:])                                                                                                  #Length of dataframe
        df=df.to_dict()

        df[Num_names[sheet]] = num                                                                                          #Store length of dataframe in dictionary
        df[List_names[sheet]] = np.arange(1,num+1)

        data[Data_names[sheet]] = df                                                                                        #Store dataframe in dictionary

        data["Reference node"] = Ref_node                                                                                   #Reference node where delta = 0

    return(data)


def OPF_model(Data):

    """
    -- Setting up the optimization model, run it and display the results. ---
    """

    model = pyo.ConcreteModel(name="(AC OPF)")

    """
    ***** Sets (lists with nodes and lines) ****
    """

    model.Lines     = pyo.Set(ordered = True, initialize = Data["Lines"]["List_of_lines"])                              #Set of AC lines
    model.Nodes     = pyo.Set(ordered = True, initialize = Data["Nodes"]["List_of_nodes"])                              #Set for nodes
    model.Time      = pyo.Set(ordered = True, initialize = Data["Time"]["List_of_times"])                               #Set for demand variability
    model.GenUnits  = pyo.Set(ordered = True, initialize = [1,2])                                                       #Set of generator units (1 in node 1 & 2 in node 2)
    model.LoadUnits = pyo.Set(ordered = True, initialize = [1,2])                                                       #Set of load units (1 in node 2 & 2 in node 3)

    """
    **** Parameters ****
    """

    model.demand_P  = pyo.Param(model.Nodes, initialize = Data["Nodes"]["DEMAND_P"])
    model.demand_Q  = pyo.Param(model.Nodes, initialize = Data["Nodes"]["DEMAND_Q"])
    model.P_min     = pyo.Param(model.Nodes, initialize = Data["Nodes"]["P_MIN"])                                        #Minimum active production at node
    model.P_max     = pyo.Param(model.Nodes, initialize = Data["Nodes"]["P_MAX"])                                        #Maximum active production at node
    model.Q_min     = pyo.Param(model.Nodes, initialize = Data["Nodes"]["Q_MIN"])                                        #Minimum reactive production at node
    model.Q_max     = pyo.Param(model.Nodes, initialize = Data["Nodes"]["Q_MAX"])                                        #Maximum reactive production at node
    model.Cost_gen  = pyo.Param(model.Nodes, initialize = Data["Nodes"]["GENCOST"])                                      #Parameter for generation cost for every node
    model.V_min     = pyo.Param(model.Nodes, initialize = Data["Nodes"]["V_MIN"])                                        #Minimum voltage at each node
    model.V_max     = pyo.Param(model.Nodes, initialize = Data["Nodes"]["V_MAX"])                                        #Maximum voltage at each node
    model.Del_min   = pyo.Param(model.Nodes, initialize = Data["Nodes"]["D_MIN"])
    model.Del_max   = pyo.Param(model.Nodes, initialize = Data["Nodes"]["D_MAX"])


    #Lines
    model.R          = pyo.Param(model.Nodes, initialize = Data["Lines"]["R"])                                             #Parameter for resistance on line
    model.X          = pyo.Param(model.Nodes, initialize = Data["Lines"]["X"])                                             #Parameter for reactance on line
    model.P_line_max = pyo.Param(model.Lines, initialize = Data["Lines"]["CAP FROM"])                                    #Parameter for maximum transfer from node, for every line.
    model.P_line_min = pyo.Param(model.Lines, initialize = Data["Lines"]["CAP TO"])
    model.line_from  = pyo.Param(model.Lines, initialize = Data["Lines"]["From"])
    model.line_to    = pyo.Param(model.Lines, initialize = Data["Lines"]["To"])


    #Demand variability

    #model.demand_var = pyo.Param(model.Time, initialize = Data["Time"]["d"])

    """
    ***** Variables *****
    """

    model.gen_P      = pyo.Var(model.Nodes)                                                                                #Variable generated active power on each node
    model.gen_Q      = pyo.Var(model.Nodes)                                                                                #Variable for generated reactive power at each node
    model.voltage    = pyo.Var(model.Nodes, initialize = 1.0, bounds = (0.9, 1.1))                                         #Variable for voltage on on bus for every node
    model.delta      = pyo.Var(model.Nodes, initialize = 0.0, bounds = (-1.5, 1.5))                                        #Variable for voltage angle on bus for every node
    #model.demand_P = pyo.Var(model.Nodes)
    #model.demand_Q = pyo.Var(model.Nodes)

    #Lines
    model.active_flow   = pyo.Var(model.Lines, bounds = (-150, 150))                                                               #Variable for power flow on each line
    model.reactive_flow = pyo.Var(model.Lines, bounds = (-150, 150))

    """
    ***** Objective function *****
    Minimize cost for production (marginal cost for generators)
    """

    def Objective_function(model):
        return (sum(model.gen_P[n]*model.Cost_gen[n] for n in model.Nodes))                                                 #Only one MC per node, need to incorporate real-time
    model.OBJ = pyo.Objective(rule = Objective_function, sense = pyo.minimize)                                              #Want to minimize generation cost

    """
    ***** Constraints *****
    """
    
    def ref_node_delta(model):
        return(model.delta[Data["Reference node"]] == 0)
    model.ref_node_const = pyo.Constraint(rule = ref_node_delta)

    def ref_node_voltage(model):
        return(model.voltage[Data["Reference node"]] == 1.05)
    model.ref_node_voltage_const = pyo.Constraint(rule = ref_node_voltage)


    """
    ***** General constraints *****
    """

    def  Min_gen_P(model,n):                                                                                                  #Generators providing more than min power
        return(model.gen_P[n] >= model.P_min[n])
    model.Min_gen_const = pyo.Constraint(model.Nodes, rule = Min_gen_P)

    def Max_gen_P(model,n):                                                                                                   #Generators providing less than max power
        return(model.gen_P[n] <= model.P_max[n])
    model.Max_gen_const = pyo.Constraint(model.Nodes, rule = Max_gen_P)

    def Min_gen_Q(model,n):
        return(model.gen_Q[n] >= model.Q_min[n])
    model.Min_gen_Q = pyo.Constraint(model.Nodes, rule = Min_gen_Q)

    def Max_gen_Q(model,n):
        return(model.gen_Q[n] <= model.Q_max[n])
    model.Max_gen_Q = pyo.Constraint(model.Nodes, rule = Max_gen_Q)

    def Max_vol(model,n):
        return(model.voltage[n] <= model.V_max[n])                                                                             #Maximum voltage
    model.Max_vol = pyo.Constraint(model.Nodes, rule = Max_vol)

    def Min_vol(model,n):
        return(model.voltage[n] >= model.V_min[n])                                                                             #Minimum voltage
    model.Min_vol = pyo.Constraint(model.Nodes, rule = Min_vol)

    def From_flow(model,l):                                                                                                 #Maximum flow on lines
        return(model.active_flow[l] <= model.P_line_max[l])
    model.From_flow_L = pyo.Constraint(model.Lines, rule = From_flow)

    def To_flow(model,l):                                                                                                   #Minimum flow on lines
        return(model.active_flow[l] >= -model.P_line_min[l])
    model.To_flow_L = pyo.Constraint(model.Lines, rule = To_flow)

    def Max_del(model,n):
        return(model.delta[n] <= model.Del_max[n])
    model.Max_del = pyo.Constraint(model.Nodes, rule = Max_del)

    def Min_del(model,n):
        return(model.delta[n] >= model.Del_min[n])
    model.Min_del = pyo.Constraint(model.Nodes, rule = Min_del)

    #Equations for active and reactive power flow on each line

    def active_flow(model, l):
        return model.active_flow[l] == (((model.voltage[l]**2)*np.cos(Theta[model.line_to[l]-1][model.line_from[l]-1])-\
                model.voltage[model.line_from[l]]*model.voltage[model.line_to[l]]*np.cos(Theta[model.line_to[l]-1][model.line_from[l]-1])) / \
               (np.sqrt((model.R[l]**2)+model.X[l]**2)))*Sbase
    model.Active_flow = pyo.Constraint(model.Lines, rule=active_flow)

    def reactive_flow(model, l):
        return model.reactive_flow[l] == (((model.voltage[l]**2)*np.sin(Theta[model.line_to[l]-1][model.line_from[l]-1])-\
                model.voltage[model.line_from[l]]*model.voltage[model.line_to[l]]*np.sin(Theta[model.line_to[l]-1][model.line_from[l]-1])) / \
               (np.sqrt((model.R[l]**2)+model.X[l]**2)))*Sbase
    model.Reactive_flow = pyo.Constraint(model.Lines, rule=reactive_flow)

    

    #def active_flow(model,l):
    #    return np.cos(model.voltage[model.line_from[l]])
    #model.active_flow = pyo.Constraint(model.Lines, rule=active_flow)

    #np.cos(model.delta[model.line_from[l] - model.delta[model.line_to[l]]])
    #model.delta[model.line_from[l]] - model.delta[model.line_to[l]]

    #Reactive flow
    #def reactive_flow(model,l):
    #    return((model.voltage[model.line_from])**2)*np.sin(Theta[model.line_to - 1][model.line_from - 1]) - \
    #          model.voltage[model.line_from] * model.voltage[model.line_to] * np.sin(model.delta[model.line_from] - \
    #          model.delta[model.line_to] - model.delta[model.line_from] + Theta[model.line_to - 1][model.line_from -1])
    #model.reactive_flow = pyo.Constraint(model.Lines, rule = active_flow)

    #Active power balance P-D = sum (P_ij))

    #def active_powbal(model,n):
    #    return(model.gen_P[n] - model.Demand_P[n] == sum(model.active_flow))
    #model.active_bal = pyo.Constraint(model.Lines, rule = active_powbal)

    #def reactive_powbal(model,n):
    #    return(model.gen_Q[n] - model.Demand_Q[n] == sum(model.reactive_flow))
    #model.reactive_flow = pyo.Constraint(model.Lines,rule = active_powbal)

    """
    ***** Compute the optimization problem ******
    """

    opt = SolverFactory("ipopt")                                                                                            #The solver used
    results = opt.solve(model, load_solutions = True)                                                                       #Solve the problem
    results.write(num=1)

    """
    Display results
    """

    model.display()                                                                                                         #Shows results


Main()                                                                                                                      #Run the main