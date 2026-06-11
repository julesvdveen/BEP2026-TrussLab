import json
import numpy as np
import matplotlib.pyplot as plt

class Beam:
    """
    Beam class to save all beam data into a single object.

    Attributes:

        L (float): inductance
        C (float): conductance
        R_0 (float): initial (reference) resistance
        filter (str): integrated filter

        length (float): initial beam length
        k (float): beam stiffness

        type (str): beam type/category
        ID (str): beam ID/name

    Methods:
        admittance(w, R): returns (complex) admittance for given frequency w.
    """

    def __init__(self, filter:str, L:float, C:float, R_0:float, length:float, k:float, type:str, ID:str):

        if filter not in ['serie', 'notch', 'parallel_1', 'parallel_2']:
            raise Exception(f'Unknown filter: {filter}')
        
        self.filter = filter
        self.L = L
        self.C = C
        self.R_0 = R_0
        self.length = length
        self.k = k
        self.type = type
        self.ID = ID

    def admittance(self, w, R=None):
        r_l = 20.9
        if R == None: R = self.R_0
        if np.any(w) == 0: return 1e-9

        if self.filter == 'serie': # All components serie
            num = self.C*w*1j
            den = -self.L*self.C*w**2 + R*self.C*w*1j + 1
        
        elif self.filter == 'notch': # all components parallel
            num = 1/R+1j*w*self.C+(r_l-1j*w*self.L)/(r_l**2+(w*self.L)**2)
            den = 1

        elif self.filter == 'parallel_1': # series(R, parallel(C, L))
            num = -self.C*self.L*w**2 + 1
            den = -R*self.C*self.L*w**2 + self.L*w*1j + R
        
        elif self.filter == 'parallel_2': # parallel(series(R, L), C)
            num = -self.L*self.C*w**2 + R*self.C*w*1j + 1
            den = self.L*w*1j + R

        return num/den
    
    def filter_plot(self, f=np.logspace(1, 5, 1000)):
        w = f * 2 * np.pi
        y = self.admittance(w)
        z_abs = 1 / np.abs(y) # Plotting Impedance is better for Notch filters

        fig, ax = plt.subplots()
        # Use loglog to see the 'V' shape clearly!
        ax.loglog(f, z_abs, color='blue', linewidth=2)
    
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_ylabel(r'Impedance Magnitude $|Z(j\omega)|$ [$\Omega$]')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_title(rf'Impedance Signature for {self.ID} ({self.filter} filter)')
    
        # Draw a line at R_0 to show the 'target'
        ax.axhline(self.R_0, color='red', linestyle='--', label=f'R_0 = {self.R_0} $\Omega$')
        ax.legend()
    
        plt.tight_layout()
        plt.show()
    
    def to_dict(self):

        beam_dict = {
            'filter': self.filter,
            'L': self.L, 'C': self.C, 'R_0': self.R_0,
            'length': self.length, 'k' : self.k,
            'type' : self.type, 'ID': self.ID 
        }
        return beam_dict


def save_beam_library(library:dict, filename:str):
    """
    Saves a beam library {ID: Beam(...)} as JSON file
    """

    data = {ID: beam.to_dict() for ID, beam in library.items()}
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f'{filename} was succesfully saved.')

def open_beam_library(filename:str):
    """
    Opens a JSON file beam library as dict.
    N.B. file originally saved using save_beam_library().
    """

    with open(filename, "r") as f:
        data = json.load(f)

    library = {ID: Beam(
        beam_dict['filter'],
        beam_dict['L'], beam_dict['C'], beam_dict['R_0'], 
        beam_dict['length'], beam_dict['k'],
        beam_dict['type'], beam_dict['ID']
        ) for ID, beam_dict in data.items()}
    
    return library

def sublibrary(str_list, library):
    """
    Returns a subset of the input library as requested per list argument.
    """

    keys = list(library.keys())
    mask = np.isin(str_list, keys)
    if not np.all(mask):
        raise Exception('Given library does not contain all requested elements')
    if np.unique(str_list).size != len(str_list):
        raise Exception('Requested beam list should contain all unique IDs.')
    
    sublib = {ID : library[ID] for ID in str_list}
    return sublib