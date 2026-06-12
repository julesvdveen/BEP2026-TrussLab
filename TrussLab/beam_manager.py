# The beam_manager: it contains all the data of the beams.

import json
import numpy as np

class Beam:
    """
    Beam class to save all beam data into a single object.

    Attributes:
        filter (str): filter topology, one of ['serie', 'notch', 'BVD', 'R'].
        L (float): inductance of the coil in Henry.
        C (list): capacitance value(s) in Farad. For 'serie' and 'notch' filters, a single value is expected. For 'BVD', provide [C_s, C_p].
        R_0 (float): initial (undeformed) resistance of the beam in Ohm.
        R_coil (float): internal coil resistance in Ohm.
        R_cap (float): internal capacitor dissipation resistance in Ohm.
        R_contact (float): contact resistance at the beam connectors in Ohm.
        l_0 (float): initial (undeformed) beam length in mm.
        alpha (float): resistivity constant in mm/Ohm, coupling elongation to resistance.
        k (float): beam stiffness in N/mm.
        type (str): beam category label (e.g. 'short', 'long').
        ID (str): unique beam identifier (e.g. 'A1', 'B2').

    Methods:
        admittance(w, R): returns (complex) admittance for given frequency w --> being used by the bridge manager.
        impedance_plot(ax, f_arr, color): returns the plot of the impedance for the given filter and properties --> can be used for more insight.
        to_dict(self): defines all the properties of the beams --> being used by the bridge manager.
    """

    def __init__(self, filter:str, L:float, C:list, R_0:float,
                  R_coil:float, R_cap:float, R_contact:float,
                    l_0:float, alpha:float, k:float,
                      type:str, ID:str):
        """
        Initialises a beam object with its filter type and physical properties.
        filters:
            A filter can be a 'serie', 'notch', 'BVD' or 'R'.
        
            A serie filter is: inductor - resistor - capacitor -
            
                                |- inductor -|
            A notch filter is:- |- resistor -| -
                                |- capacitor-|

            A BVD filter is: - inductor - resistor  - capacitor -
                                        | capacitor |
            
            A R filter  is   - resistor - 
        """

        if filter not in ['serie', 'notch', 'BVD', 'R']:
            raise Exception(f'Unknown filter: {filter}')
        
        self.filter = filter
        self.L = L

        if filter == 'serie' or filter == 'notch':
            self.C = C

        else:
            if len(C) != 2:
                raise Exception('BVD filter was selected, but the capacitance list argument does not have a length of two ([C_s, C_p]).')
            self.C = C
        
        self.R_0 = R_0
        self.R_coil = R_coil
        self.R_cap = R_cap
        self.R_contact = R_contact

        self.l_0 = l_0
        self.alpha = alpha

        self.k = k

        self.type = type
        self.ID = ID

    def admittance(self, w, R=None):
        """
        admittance(w, R): returns (complex) admittance for given frequency w --> being used in the bridge manager.
        """
        if R is None: R = self.R_0
        if np.any(w) == 0: return 1e-9

        if self.filter == 'serie': # All components serie
            num = self.C*w*1j
            den = -self.L*self.C*w**2 + (R+self.R_loss)*self.C*w*1j + 1
        
        elif self.filter == 'notch': # all components parallel
            R_coil = self.R_coil      # serieweerstand van de spoel
            R_cap = self.R_cap        # serieweerstand van de condensator (ESR)
            R_contact = self.R_contact
            
            Z_L = R_coil + self.L * w * 1j
            Z_C = R_cap/(self.C*w) + 1 / (self.C * w * 1j)
            Z_R = R

            # Parallelle admittance van drie takken:
            Y = 1/Z_L + 1/Z_C + 1/Z_R
            
            # Impedantie inclusief serie R_loss:
            Z_totaal = 1/Y + R_contact
            
            return 1/Z_totaal

        elif self.filter == 'BVD': # Butterworth-Van Dyke
            C_s = self.C[0]
            C_p = self.C[1]
            R_loss = self.R_loss
            num = -self.L*C_s*C_p*w**3*1j - (R+R_loss)*C_s*C_p*w**2 + (C_s + C_p)*w*1j
            den = -self.L*C_s*w**2 + (R+R_loss)*C_s*w*1j + 1

        elif self.filter == 'R': # Single resistor for reference testing.
            num = 1
            den = R
        
        return num/den
    
    def impedance_plot(self, ax, f_arr, color):
        """
        impedance_plot(ax, f_arr, color): returns the plot of the impedance for the given filter and properties --> can be used for more insight.
        """
        w_arr = f_arr * 2 * np.pi
        y = self.admittance(w_arr)
        z_abs = 1 / np.abs(y)
        ax.semilogx(f_arr, z_abs, color=color, label = self.ID)
    
    def to_dict(self):
        """
        to_dict(self): defines all the properties of the beams --> being used in the bridge manager.
        """
        beam_dict = {
            'filter': self.filter,
            'L': self.L, 'C': self.C, 'R_0': self.R_0,
            'R_coil': self.R_coil, 'R_cap': self.R_cap, 'R_contact': self.R_contact,
            'l_0': self.l_0, 'alpha': self.alpha, 'k' : self.k,
            'type' : self.type, 'ID': self.ID 
        }
        return beam_dict


def save_beam_library(library:dict, filename:str):
    """
    Saves a beam library {ID: Beam(...)} as JSON file. --> being used in the 'beam_init' file
    """

    data = {ID: beam.to_dict() for ID, beam in library.items()}
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f'{filename} was succesfully saved.')

def open_beam_library(filename:str):
    """
    Opens a JSON file beam library as dict.
    N.B. file originally saved using save_beam_library(). --> being used by the bridge_manager.
    """

    with open(filename, "r") as f:
        data = json.load(f)

    library = {ID: Beam(
        beam_dict['filter'],
        beam_dict['L'], beam_dict['C'], beam_dict['R_0'], 
        beam_dict['R_coil'], beam_dict['R_cap'], beam_dict['R_contact'],
        beam_dict['l_0'], beam_dict['alpha'], beam_dict['k'],
        beam_dict['type'], beam_dict['ID']
        ) for ID, beam_dict in data.items()}
    
    return library

def sublibrary(str_list, library):
    """
    Returns a subset of the input library as requested per list argument. --> Being used by the bridge_manager
    """

    keys = list(library.keys())
    mask = np.isin(str_list, keys)
    if not np.all(mask):
        raise Exception('Given library does not contain all requested elements')
    if np.unique(str_list).size != len(str_list):
        raise Exception('Requested beam list should contain all unique IDs.')
    
    sublib = {ID : library[ID] for ID in str_list}
    return sublib