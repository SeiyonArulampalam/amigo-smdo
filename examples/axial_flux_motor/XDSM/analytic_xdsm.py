from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT, RIGHT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=False)

# Define blocks
x.add_system("Stator_Geometry", FUNC, r"\text{Stator Geometry}")
x.add_system("Steinmetz", FUNC, r"\text{Steinmetz Equation}")
x.add_system("Thermal", FUNC, r"\text{Thermal Resistance Network}")
x.add_system("Motor_Sizing_Equation", FUNC, r"\text{Motor Sizing Equation}")

x.add_input("Stator_Geometry", r"\text{Design Variables}")
x.add_input("Motor_Sizing_Equation", r"\text{Requirements}")
x.connect("Stator_Geometry", "Steinmetz", r"f,B,V_{\text{Fe}}")
x.connect("Steinmetz", "Thermal", r"P_{\text{Fe}}")
x.connect("Thermal", "Motor_Sizing_Equation", r"P_{\text{Cu}},A")
x.add_output("Motor_Sizing_Equation", r"P_{\text{out}}", side=RIGHT)

x.write("analytic_xdsm", quiet=True)

# Design variables
# -calculate stator geometry
# -calculate iron loss
# -calculate copper loss
