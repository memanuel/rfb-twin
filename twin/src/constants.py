# *************************************************************************************************
# Physical constants
R: float = 8.3145       # Ideal gas constant in J / (mol K)
T: float = 298.15       # Room temperature in Kelvin
F: float = 96485.3329   # Faraday's constant in Coulombs per mole
V_T: float = R*T*1E3/F  # Thermal voltage in millivolts
ne: int = 2             # number of electrons
c0: float = 20.0E-3     # Total AQDS concentration in moles per liter
diff: float = 4.0E-6    # The diffusivity of AQDS in cm^2 / sec
k0: float = 7.2E-3      # The kinetic rate constant of AQDS in cm / sec
