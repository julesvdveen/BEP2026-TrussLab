import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from scipy.ndimage import median_filter  

# filename = "resultaten1.txt"
# f_start = np.array([1000])
# f_stop = np.array([20000])
# steps = np.array([1500])
VI_PATH = str(Path(__file__).parent / "temp" / "Mydaq12.vi")
# VI_PATH = r"C:\Users\Reinoud van Holk\OneDrive - Delft University of Technology\Bestanden van Jules van der Veen - BEP 2026\Python\U-U Filter\temp\Mydaq12.vi"
#" C:\Data\aansturen mydaq\Mydaq123.vi"

def logs_to_U(filename_log:str, U_in:float=1, Cartesian=True):

    """
    Converts myDAQ Bode log to frequency and complex voltage arrays.
    """
    df = pd.read_csv('myDAQ_logs/' + filename_log, skiprows=3, sep='\s+', decimal=',', 
                 names=['freq', 'gain_db', 'phase_deg'])
    
    f_arr = df['freq'].values

    U_gain = 10**(df['gain_db'].values/20)
    U_phase = df['phase_deg'].values

    if not Cartesian:
        return f_arr, U_gain, U_phase
    
    else:
        U_phase_rad = np.radians(U_phase)
        U = U_in*U_gain*(np.cos(U_phase_rad) + 1j*np.sin(U_phase_rad))
        return f_arr, U


def myDAQ_sweep(f_start_list:list, f_stop_list:list, steps:list, filename:str, Cartesian:bool=True):
    print("myDAQsweep started...")
    Path("temp").mkdir(exist_ok=True)
    import win32com.client

    """
    Function to operate the myDAQ manager from Python, starts a labview file to sweep the bridge.

    Args:
    f_start (list): starting freq
    f_stop (list): stopping freq
    steps (list): steps/decade
    """

    if len(f_start_list) != len(f_stop_list) or len(f_start_list) != len(steps):
        raise Exception("The start, stop and steps lists do not have matching dimensions.")
    
    labview = win32com.client.Dispatch("LabVIEW.Application")
    vi = labview.GetVIReference(VI_PATH)

    # Belangrijk: pywin32 moet weten dat dit methodes zijn
    vi._FlagAsMethod("SetControlValue")
    vi._FlagAsMethod("Run")

    vi.FPWinOpen = True

    f_arr = []
    U_gain = []
    U_phase = []
    
    for start_frequency_Hz, stop_frequency_Hz, steps_per_decade in zip(f_start_list, f_stop_list, steps):
        vi.SetControlValue("start_frequency_Hz", float(start_frequency_Hz))
        vi.SetControlValue("stop_frequency_Hz", float(stop_frequency_Hz))
        vi.SetControlValue("steps_per_decade", int(steps_per_decade))


        vi.Run(True)
        Path(__file__).parent / "temp" / "test.lvm"

        while not lvm.exists():
            time.sleep(1)

        #Uitlezen

        df = pd.read_csv(lvm, sep="\t", decimal=",", skiprows=23, usecols=[1, 2, 3], names=["frequency", "gain", "phase"])
        df = df.dropna()
        df = df.replace(",", ".", regex=True).astype(float)

        frequency_i = df["frequency"]
        U_gain_i = 10**(df['gain'].values/20)
        U_phase_i = df['phase'].values

        f_arr.extend(frequency_i)
        U_gain.extend(U_gain_i)
        U_phase.extend(U_phase_i)

        lvm = Path(__file__).parent / "temp" / "test.lvm"

        if lvm.exists():
            lvm.unlink()

        print(f"Sweeping range {start_frequency_Hz} to {stop_frequency_Hz} is finished.")

    f_arr = np.array(f_arr)
    U_gain = np.array(U_gain)
    U_gain_filtered = median_filter(U_gain, size=3)
    U_phase = np.array(U_phase)
    U_phase_filtered = median_filter(U_phase, size=3)

    tijd_datum = datetime.now().strftime("%d-%m-%Y\t%H:%M")
    data = np.column_stack((f_arr, 20 * np.log10(U_gain_filtered), U_phase_filtered))
    header = f"{tijd_datum}\nAmplitude: 2,00 V\nFreq (Hz)\tGain (dB)\tPhase (deg)"
    np.savetxt(filename, data, header=header, comments="", fmt="%.3f", delimiter="\t")
    
    with open(filename, "r", encoding="utf-8") as f:
        txt = f.read()

    txt = txt.replace(".", ",")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(txt)

    if not Cartesian:
        return f_arr, U_gain_filtered, U_phase_filtered
        
    else:
        U_phase_rad = np.radians(U_phase_filtered)
        U = U_gain_filtered*(np.cos(U_phase_rad) + 1j*np.sin(U_phase_rad))
        return f_arr, U



