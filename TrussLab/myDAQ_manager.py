# The myDAQ_manager is able to control the myDAQ. The python file controls labVIEW and labVIEW controls the myDAQ.
# The code below opens a labVIEW file on your laptop and runs it.
# It is important for the user to have all the folders on your laptop sequenced the same way as in Python.

import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from scipy.ndimage import median_filter  

VI_PATH = str(Path(__file__).parent / "temp" / "Mydaq12.vi")

def logs_to_U(filename_log:str, U_in:float=1, Cartesian=True):
    """
    Converts myDAQ Bode log to frequency and complex voltage arrays. 
    This code does not run labVIEW, so you will have to sweep the frequencies
    manually using the 'National Instruments' app.
    """

    df = pd.read_csv(filename_log, skiprows=3, sep='\s+', decimal=',', 
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
    """
    Function to operate the myDAQ manager from Python, starts a labview file to sweep the bridge.

    Arguments:
    f_start (list): starting freq
    f_stop (list): stopping freq
    steps (list): steps/decade

    Explanation:
    This function is capable to start and run a labVIEW file. 
    This function is capable of doing the information dense frequency sweeping, 
        because it deletes the previous 'test' txt-files. 
    This function has a loop of a time delay, 
        because the code needs to wait for the myDAQ till it is done sweeping.
    When labVIEW is done sweeping, it copies the values to a list. The list is converted to a txt-file.
    """
    import win32com.client # Only compatible with Windows

    print("myDAQsweep started...")
    Path("temp").mkdir(exist_ok=True)
    
    if len(f_start_list) != len(f_stop_list) or len(f_start_list) != len(steps):
        raise Exception("The start, stop and steps lists do not have matching dimensions.")
    
    labview = win32com.client.Dispatch("LabVIEW.Application")
    vi = labview.GetVIReference(VI_PATH)

    # Important: pywin32 needs to know that these are methods:
    vi._FlagAsMethod("SetControlValue")
    vi._FlagAsMethod("Run")

    vi.FPWinOpen = True

    f_arr = []
    U_gain = []
    U_phase = []

    lvm = Path(__file__).parent / "temp" / "test.lvm"

    if lvm.exists():
        lvm.unlink()
    

    for start_frequency_Hz, stop_frequency_Hz, steps_per_decade in zip(f_start_list, f_stop_list, steps):
        vi.SetControlValue("start_frequency_Hz", float(start_frequency_Hz))
        vi.SetControlValue("stop_frequency_Hz", float(stop_frequency_Hz))
        vi.SetControlValue("steps_per_decade", int(steps_per_decade))

        vi.Run(True) # This runs the labVIEW file 
        lvm = Path(__file__).parent / "temp" / "test.lvm"

        # Record the starting time of the loop
        start_time = time.time()
        timeout_seconds = 300 # s

        while not lvm.exists():
            # Check if the elapsed time exceeds the timeout limit
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"LabVIEW measurement timed out! 'test.lvm' was not created within {timeout_seconds} seconds."
                )
            
            time.sleep(1)

        #Read
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

    # This filter is able to filter out the outliers of the measurements.
    f_arr = np.array(f_arr)
    U_gain = np.array(U_gain)
    U_gain_filtered = median_filter(U_gain, size=5)
    U_phase = np.array(U_phase)
    U_phase_filtered = median_filter(U_phase, size=5)

    # This makes from the list a txt-file in the same way txt-files are made by 'National Instruments'
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



