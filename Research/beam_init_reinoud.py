from beam_manager import Beam, save_beam_library

# For 5000 Ohm simulations, we need higher L to keep the Q-factor high.
# We aim for Z0 (sqrt(L/C)) to be roughly 25k-50k Ohm.

#R_cap = D/C
test_library = {
    'A1': Beam('notch', L=9.569e-3, C=[10.079e-9], R_0=536, R_coil=20.02, R_cap=0.03, R_contact=0, l_0=1.0, alpha=1, k=900.0, type='A', ID='A1'),
    'A2': Beam('notch', L=10.0e-3, C=[22e-9], R_0=4.7e2, R_coil=20.0, R_cap=0.03, R_contact=0.0, l_0=1.2, alpha=1, k=850.0, type='A', ID='A2'),

    #'B1': Beam('notch', L=9.605e-3, C=[87.2e-9], R_0=4.68e2, R_coil=21.13, R_cap=0.05, R_contact=5.0, length=1.5, k=700.0, type='B', ID='B1'),
    'B1': Beam('notch', L=9.482e-3, C=[85.7e-9], R_0=521, R_coil=21.64, R_cap=0.03, R_contact=5.0, l_0=1.5, alpha=1, k=700.0, type='B', ID='B1'),
    'B2': Beam('notch', L=21.94e-3, C=[87.3e-9], R_0=533, R_coil=41.55, R_cap=0.03, R_contact=5.0, l_0=1.8, alpha=1, k=600.0, type='B', ID='B2'),
    # --- TYPE C: High-Frequency (~7.0 kHz - 9.5 kHz) ---
    'B3': Beam('notch', L=9.5e-3, C=[86e-9], R_0=388, R_coil=21, R_cap=0.03, R_contact=0.0, l_0=2.0, alpha=1, k=550.0, type='B', ID='B3'),
    'C1': Beam('notch', L=20.0e-3, C=[470e-9], R_0=4.7e2, R_coil=40.0, R_cap=0.12, R_contact=5.0, l_0=2.2, alpha=1, k=450.0, type='C', ID='C1'),
    'C2': Beam('notch', L=20.0e-3, C=[1.0e-6], R_0=4.7e2, R_coil=40.0, R_cap=0.18, R_contact=5.0, l_0=2.5, alpha=1, k=350.0, type='C', ID='C2')
}

save_beam_library(test_library, 'notch4test.json')

#For notch: The resistance does matter. But the correlation is very difficult to find.

#Mogelijke condensatoren
#1e-4 elektro 100microF --
#1e-5 elektro   10 microF
#1e-7 keramisch 104 --> 0.93077e-7
#1e-8 keramisch 103
#1e-9 keramisch 102
#1e-10 keramisch 101
#1e-11 keramisch 100

#Mogelijke inductors
#0.01
# 226C eig: 0.0211

#Volgens AI is de hoogste frequentie die erdoor heen kan 20kHz
#LC is op zijn laagst 6e-11
#Het is redelijk tot LC = 2.5e-10


