from beam_manager import Beam, save_beam_library
import numpy as np

# LCR meting gekochte library (default)
library = {
    'S1': Beam('notch', L=9.623e-3, C=[9.892e-9], R_0=5.030e2, R_coil=20.36, R_cap=0.008, R_contact=5.0, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S1'), # R_serie = 218.2 Ohm
    'S2': Beam('notch', L=9.569e-3, C=[20.33e-9], R_0=5.135e2, R_coil=20.08, R_cap=0.012, R_contact=5.0, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S2'), # R_serie = 218.6 Ohm
    'S3': Beam('notch', L=9.590e-3, C=[87.00e-9], R_0=4.860e2, R_coil=20.22, R_cap=0.013, R_contact=5.0, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S3'), # R_serie = 218.7 Ohm
    'S4': Beam('notch', L=22.42e-3, C=[84.49e-9], R_0=4.659e2, R_coil=42.69, R_cap=0.014, R_contact=5.0, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S4'), # R_serie = 218.7 Ohm
    
    'L1': Beam('notch', L=22.13e-3, C=[223.2e-9], R_0=8.180e2, R_coil=42.18, R_cap=0.015, R_contact=5.0, l_0=100.0*np.sqrt(2), alpha=.05, k=1.92, type='L', ID='L1'), # R_serie = 217.6 Ohm
    'L2': Beam('notch', L=22.13e-3, C=[473.3e-9], R_0=8.169e2, R_coil=41.77, R_cap=0.015, R_contact=5.0, l_0=100.0*np.sqrt(2), alpha=.05, k=1.92, type='L', ID='L2'), # R_serie = 217.7 Ohm
    'L3': Beam('notch', L=22.13e-3, C=[855.2e-9], R_0=8.630e2, R_coil=42.84, R_cap=0.018, R_contact=5.0, l_0=100.0*np.sqrt(2), alpha=.05, k=1.92, type='L', ID='L3')
}

# 08-06 gekalibreerde meting gekochte library (default)
# library = {
#     'S1': Beam('notch', L=9.623e-3, C=[9.892e-9], R_0=5.030e2, R_coil=20.36, R_cap=0.08, R_contact=.5, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S1'), # R_serie = 218.2 Ohm
#     'S2': Beam('notch', L=9.569e-3, C=[20.33e-9], R_0=5.135e2, R_coil=20.08, R_cap=0.012, R_contact=.5, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S2'), # R_serie = 218.6 Ohm
#     'S3': Beam('notch', L=9.590e-3, C=[87.00e-9], R_0=4.860e2, R_coil=20.22, R_cap=0.013, R_contact=.5, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S3'), # R_serie = 218.7 Ohm
#     'S4': Beam('notch', L=22.42e-3, C=[84.49e-9], R_0=4.659e2, R_coil=42.69, R_cap=0.014, R_contact=.5, l_0=100.0, alpha=.05, k=1.92, type='S', ID='S4'), # R_serie = 218.7 Ohm
    
#     'L1': Beam('notch', L=22.13e-3, C=[223.2e-9], R_0=8.180e2, R_coil=42.18, R_cap=0.015, R_contact=.5, l_0=100.0*np.sqrt(2), alpha=.05, k=1.92, type='L', ID='L1'), # R_serie = 217.6 Ohm
#     'L2': Beam('notch', L=22.13e-3, C=[473.3e-9], R_0=8.169e2, R_coil=41.77, R_cap=0.015, R_contact=.5, l_0=100.0*np.sqrt(2), alpha=.05, k=1.92, type='L', ID='L2'), # R_serie = 217.7 Ohm
#     'L3': Beam('notch', L=22.13e-3, C=[855.2e-9], R_0=8.630e2, R_coil=42.84, R_cap=0.018, R_contact=.5, l_0=100.0*np.sqrt(2), alpha=.05, k=1.92, type='L', ID='L3')
# }

# Theoretische gekochte library (default1)
# library = {
#     'S1': Beam('notch', L=9.623e-3, C=[9.892e-9], R_0=5.03e2, R_coil=20.36, R_cap=0.008, R_contact=5.0, l_0=1.0, alpha=.1, k=100, type='S', ID='S1'), # R_serie = 218.2 Ohm
#     'S2': Beam('notch', L=9.569e-3, C=[20.33e-9], R_0=5.135e2, R_coil=20.08, R_cap=0.012, R_contact=5.0, l_0=1.0, alpha=.1, k=100, type='S', ID='S2'), # R_serie = 218.6 Ohm
#     'S3': Beam('notch', L=9.590e-3, C=[87.00e-9], R_0=4.86e2, R_coil=20.22, R_cap=0.013, R_contact=5.0, l_0=1.0, alpha=.1, k=100, type='S', ID='S3'), # R_serie = 218.7 Ohm
#     'S4': Beam('notch', L=22.42e-3, C=[84.49e-9], R_0=4.659e2, R_coil=42.69, R_cap=0.014, R_contact=5.0, l_0=1.0, alpha=.1, k=100, type='S', ID='S4'), # R_serie = 218.7 Ohm
    
#     'L1': Beam('notch', L=22.13e-3, C=[223.2e-9], R_0=8.18e2, R_coil=42.18, R_cap=0.015, R_contact=5.0, l_0=1.0, alpha=.1, k=100, type='L', ID='L1'), # R_serie = 217.6 Ohm
#     'L2': Beam('notch', L=22.13e-3, C=[473.3e-9], R_0=8.169e2, R_coil=41.77, R_cap=0.017, R_contact=5.0, l_0=1.0, alpha=.1, k=100, type='L', ID='L2'), # R_serie = 217.7 Ohm
#     'L3': Beam('notch', L=20.0e-3, C=[1.0e-6], R_0=5.0e2, R_coil=40.0, R_cap=0.018, R_contact=5.0, l_0=1.0, alpha=.1, k=100, type='L', ID='L3')
# }

save_beam_library(library, 'default.json')

# Weerstanden los:
# 100 (A1): 99.2 Ohm (R_ref)
# 100 (B1): 98.4 Ohm (R_ref)
# 100 (B2): 98.6 Ohm (R_ref)
# 220: 218.9 Ohm
# 330: 324.4 Ohm
# 470: 464.8 Ohm
# 680: 665.3 Ohm

# Potmeter A1 (LCR: zwarte kabel linksvoor, rode kabel rechtsachter):
# Mid-left: 250 Ohm
# Mid: 536 Ohm
# Mid-right: 751 Ohm

# Potmeter B1 (LCR: zwarte kabel linksvoor, rode kabel rechtsachter):
# Mid-left: 252 Ohm
# Mid: 521 Ohm
# Mid-right: 753 Ohm

# Potmeter B2 (LCR: zwarte kabel linksvoor, rode kabel rechtsachter):
# Mid-left: 253 Ohm
# Mid: 533 Ohm
# Mid-right: 753 Ohm


# Notch filter equations:
# Q = w_o * L / R_coil = 1/R_coil * sqrt(L/C)
# f_o = 1/(2*pi*sqrt(L*C))
