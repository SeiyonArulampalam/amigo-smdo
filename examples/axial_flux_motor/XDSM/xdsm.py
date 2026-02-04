from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT, RIGHT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=False)

# Define blocks
# x.add_system("opt", OPT, r"\text{Optimizer}")
x.add_system("morph", FUNC, r"\text{Mesh Morph FEA}")
x.add_system("maxwell", FUNC, r"\text{Maxwell FEA}")
x.add_system("flux", FUNC, r"\text{Flux}")
x.add_system("performance", FUNC, r"\text{Performance}")

x.add_input("morph", r"\text{Design Variables}")
x.connect("morph", "maxwell", r"\text{Node Coords.}")
x.connect("maxwell", "flux", r"\text{Magnetic Potential (A)}")
x.connect("flux", "performance", r"\text{Magnetic Field (B)}")
x.add_output("performance", r"\eta, P_{out}, \tau", side=RIGHT)
# x.connect("morph", "opt", r"\text{mass}")

x.write("fea_xdsm", quiet=True)
