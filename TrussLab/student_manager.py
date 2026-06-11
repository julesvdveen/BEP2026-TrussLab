from .bridge_manager import Bridge
from .beam_manager import open_beam_library, sublibrary
from IPython.display import HTML, display
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import inv
import sympy as sp

# ------------------------------------------------------------------------------------
#                                 DEFAULT SETTINGS:
# ------------------------------------------------------------------------------------

R_ref = 99.61 # Ohm (Integrated into final product)
library_name = 'default'

# ------------------------------------------------------------------------------------

class project:

    def __init__(self):

        self.struc = Bridge()

        self.K_trans = None
        self.K_gl = None
        self.K_local = None

        self.K_RR = None
        self.K_RG = None
        self.K_GR = None
        self.K_GG = None

        self.dR_indices = None
        self.dG_indices = None

        self.d_full = None
        self.dG = None
        self.dR = None

        self.f_full = None
        self.fG = None
        self.fR = None

        self.x_mes = None
        self.R_mes = None
        self.u_mes = None

    def construct(self, selection:list, topology:list, rigid:int=0):
        """
        """

        self.__init__()

        library = open_beam_library(library_name + '.json')
        lib = sublibrary(selection, library)

        self.struc.construct(lib, topology, rigid)
        self.K_local_transformed()
        self.K_global()


    def properties(self):
        beams = list(self.struc.library.values())

        table_data = []
        for beam in beams:
            table_data.append([beam.ID, beam.type, "192", f"{round(beam.l_0, 1)}"])

        fig, ax = plt.subplots(figsize=(6, len(table_data) * 0.5 + 1))
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=["ID", "Category", "EA (N)", "L (mm)"],
            cellLoc="center",
            colLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#82b1fe")
            else:
                cell.set_facecolor("white")

        plt.tight_layout()
        plt.show()

    def K_local_transformed(self):

        K_transformed = []
        K_local_list = []

        beams = list(self.struc.library.values())
        tops = self.struc.topology
        coords_init = self.struc.x_init.reshape(-1,2)

        for top, beam in zip(tops, beams):

            i,j = top
            xi, yi = coords_init[i]
            xj, yj = coords_init[j]

            L = beam.l_0
            c = (xj - xi) / L
            s = (yj - yi) / L

            K_local = beam.k * np.array([[1, -1],[-1, 1]])

            T = np.array([[c, s, 0, 0],[0, 0, c, s]])


            K_global = T.T @ K_local @ T

            K_transformed.append(K_global)
            K_local_list.append(K_local)

        self.K_local = K_local_list
        self.K_trans = K_transformed # list
    
    def K_global(self):
        n_dof =2 * self.struc.n
        Kglobal = np.zeros((n_dof, n_dof))
        beams = list(self.struc.library.values())
        tops = self.struc.topology
        coords_init = self.struc.x_init.reshape(-1,2)
        for top, beam in zip(tops, beams):
            i,j = top
            xi, yi = coords_init[i]
            xj, yj = coords_init[j]
            L = beam.l_0
            c = (xj - xi) / L
            s = (yj - yi) / L

            K_local = beam.k * np.array([[1, -1],[-1, 1]])

            T = np.array([[c, s, 0, 0],[0, 0, c, s]])
            K_global = T.T @ K_local @ T
            dof = [2*i, 2*i+1, 2*j, 2*j+1]

            for ii, di in enumerate(dof):
                for jj, dj in enumerate(dof):
                    Kglobal[di, dj] += K_global[ii, jj]
            
        self.K_gl = Kglobal 

    def theory(self, nodes, force, anim: bool = False, table: bool = False):
        """
        Solves the truss system by partitioning the global matrix equation based on boundary 
        conditions, using a uniform or independent directional force magnitude.

        Parameters:
        -----------
        nodes : int or tuple/list of len() == 2
            The target node(s) where the pulley forces are applied. 
            - If an int: Both X and Y pulley lines connect to this single node.
            - If a tuple/list: nodes[0] gets the X force, nodes[1] gets the Y force.
        force : float, list, tuple, or np.ndarray
            The applied force magnitude(s) in Newtons.
            - If a single float/int: Applied equally to both active X and Y pulley strings.
            - If a sequence of len()==2: Index 0 is the X force magnitude, Index 1 is the Y force magnitude.
        """
        # 1. Validate and Parse Target Nodes
        if isinstance(nodes, (tuple, list)):
            if len(nodes) != 2:
                raise ValueError("The 'nodes' sequence argument must contain exactly 2 target node indices [node_x, node_y].")
            target_x_node, target_y_node = nodes[0], nodes[1]
        elif isinstance(nodes, int):
            target_x_node, target_y_node = nodes, nodes
        else:
            raise TypeError("The 'nodes' argument must be an integer or a sequence (tuple/list) of 2 integers.")

        # 2. New Validation Logic: Parse Force Parameter
        if isinstance(force, (list, tuple, np.ndarray)):
            if len(force) != 2:
                raise ValueError("If 'force' is specified as a sequence, it must contain exactly 2 items: [force_x, force_y].")
            fx_mag, fy_mag = force[0], force[1]
        elif isinstance(force, (int, float, np.number)):
            fx_mag, fy_mag = float(force), float(force)
        else:
            raise TypeError("The 'force' argument must be a number or a sequence (list/tuple) of 2 numbers.")

        K = self.K_gl
        n_dof = 2 * self.struc.n
        last_node = self.struc.n - 1
        dG_indices = [0, 1]
        coords_init = self.struc.x_init.reshape(-1, 2)

        if self.struc.rigid:
            dG_indices.extend([2 * last_node, 2 * last_node + 1])
        else:
            dG_indices.append(2 * last_node + 1)
        
        dR_indices = [i for i in range(n_dof) if i not in dG_indices]
        self.dR_indices = dR_indices
        self.dG_indices = dG_indices

        # Updated internal helper function to map explicit directional forces
        def apply_forces(n_dof, nx, ny, fx, fy):
            f_full = np.zeros(n_dof)
            
            # Apply individual components to their respective nodes
            f_full[2 * nx] = fx
            f_full[2 * ny + 1] = fy
            
            return f_full
        
        # Build global load vector using parsed components
        f_full = apply_forces(n_dof, target_x_node, target_y_node, fx_mag, fy_mag)
        
        # --- Remaining partitioning, sorting, and plotting code stays exactly the same ---
        fR = f_full[dR_indices]
        self.fR = fR

        K_RR = K[np.ix_(dR_indices, dR_indices)]  
        K_RG = K[np.ix_(dR_indices, dG_indices)]  
        K_GR = K[np.ix_(dG_indices, dR_indices)]  
        K_GG = K[np.ix_(dG_indices, dG_indices)]  


        dG = np.zeros(len(dG_indices))
        dR = np.linalg.inv(K_RR) @ (fR - K_RG @ dG)
        self.dR = dR
        fG = K_GR @ dR + K_GG @ dG

        d_full = np.zeros(n_dof)
        d_full[dR_indices] = dR
        d_full[dG_indices] = dG
        self.d_full = d_full
        self.dG = dG
            
        F_full = np.zeros(n_dof)
        F_full[dR_indices] = fR
        F_full[dG_indices] = fG
        self.f_full = F_full
        
        if table is True:
            print("\nRESULTATEN:")
            print(f"{'Node':<6} {'ux (mm)':<12} {'uy (mm)':<12} {'fx (N)':<12} {'fy (N)':<12}")
            print("-" * 54)
            for node in range(self.struc.n):
                print(f"{node:<6} {d_full[2*node]:<12.6f} {d_full[2*node+1]:<12.6f} "
                      f"{F_full[2*node]:<12.4f} {F_full[2*node+1]:<12.4f}")

        displacements = d_full.reshape(-1, 2)
        x_deformed = coords_init + displacements

        self.plot(x=x_deformed.flatten(), anim=anim)

        self.K_RR = K_RR 
        self.K_RG = K_RG
        self.K_GR = K_GR
        self.K_GG = K_GG
        
    def plot(self, x:list=None, anim:bool=False):

        if x is None:
            deformed = np.any(self.struc.current_x != self.struc.x_init)
        else:
            deformed = np.any(x != self.struc.x_init)

        if anim and not deformed:
            print('Animated results are not available for the undeformed structure.')
            self.struc.x_plot(x)
        elif anim and deformed:
            anim = self.struc.x_plot(x, anim=True)
            display(HTML(anim.to_jshtml()))
        else:
            self.struc.x_plot(x)

    def measure(self, save_filename, settings:dict=None, anim:bool=False):

        default_settings = { 
            'freqsteps': self.struc.k * 100,
            'resonance_sweep': True, 
            'geo_constraint': True,
            'max_iter': 500
        }

        if settings is None:
            settings = default_settings
        else:
            default_settings.update(settings)
            settings = default_settings

        freqsteps       = settings['freqsteps']
        resonance_sweep = settings['resonance_sweep']
        geo_constraint  = settings['geo_constraint']
        max_iter         = settings['max_iter']

        R_mes, x_mes = self.struc.sweep(R_ref, freqsteps=freqsteps, myDAQ=True, myDAQlog=None, save_filename=save_filename,
                                         resonance_sweep=resonance_sweep, geo_constraint=geo_constraint, max_iter=max_iter)     
        self.R_mes = R_mes
        self.struc.current_x = x_mes
        self.x_mes = x_mes

        u_mes = x_mes - self.struc.x_init
        self.u_mes = u_mes

        self.plot(anim=anim)

    def local_K(self, decimal_places: int = 2):
        """
        Retrieves the calculated local stiffness matrices for all currently selected 
        elements and displays them side-by-side using the HTML helper pipeline.
        """
        display_payload = {}
        beams = list(self.struc.library.values())
        K_loc = [[1, -1], [-1, 1]]

        for beam in beams:
            k = beam.k # stiffness (float)
            ID = beam.ID # beam ID (str)

            scalar_prefix = f"{float(k):.{decimal_places}f}".rstrip('0').rstrip('.')
            display_payload[ID] = (K_loc, scalar_prefix)
        
        self.display_matrices(display_payload, decimal_places=decimal_places)

    def f_theory(self, nodes, force: float):
        """
        Runs the system solver, constructs a symbolic global force vector 'f' 
        containing numerical active loads and unknown boundary reaction variables, 
        and displays it cleanly via the HTML matrix pipeline.
        """
        # 1. Run the structural solver to populate the internal indices and partitions
        self.theory(nodes=nodes, force=force, anim=False, table=False)
        
        n_dof = 2 * self.struc.n
        f_symbolic = [0] * n_dof
        
        # 2. Fill in the known applied forces from the solver's load vector
        for idx in self.dR_indices:
            f_symbolic[idx] = self.f_full[idx]
            
        # 3. FIX: Create clean HTML strings instead of SymPy symbols
        for idx in self.dG_indices:
            node_num = idx // 2
            direction = "x" if idx % 2 == 0 else "y"
            # Format using standard italics and subscripts just like the rest of the UI
            f_symbolic[idx] = f"<i>F</i><sub>{direction}{node_num}</sub>"
            
        # 4. Pass the column vector layout data directly to display_matrices
        f_vector_data = [[item] for item in f_symbolic]
        
        display_payload = {
            "Global Force Vector (f)": f_vector_data
        }
        
        self.display_matrices(display_payload, decimal_places=1)

    def d_theory(self):
        """
        Constructs the symbolic global displacement vector 'd' using the cached 
        solver results from f_theory(), replacing unknown displacements with symbols 
        and displaying it via the HTML matrix pipeline.
        """
        # Ensure that a simulation has actually run first to populate the attributes
        if self.dG_indices is None or self.dR_indices is None:
            raise ValueError("No system state found. Please run demo.f_theory() before viewing the displacement vector.")
            
        n_dof = 2 * self.struc.n
        d_symbolic = [0] * n_dof
        
        # 1. Fill in the known boundary displacements (e.g., 0 for rigid nodes)
        for idx in self.dG_indices:
            d_symbolic[idx] = self.d_full[idx]
            
        # 2. Fill in the unknown free displacements with clean HTML symbolic variables
        for idx in self.dR_indices:
            node_num = idx // 2
            direction = "x" if idx % 2 == 0 else "y"
            # Format using standard italics and subscripts just like the force vector
            d_symbolic[idx] = f"<i>u</i><sub>{direction}{node_num}</sub>"
            
        # 3. Convert the flat list into a column vector layout (N_dof x 1 list of lists)
        d_vector_data = [[item] for item in d_symbolic]
        
        # 4. Package it into our standard dictionary payload format
        display_payload = {
            "Global Displacement Vector (d)": d_vector_data
        }
        
        # 5. Render using your updated CSS-border HTML layout helper
        self.display_matrices(display_payload, decimal_places=1)


    def display_matrices(self, items: dict, decimal_places: int = 1):
        """
        Displays a series of matrices side-by-side or stacked using highly compatible
        raw HTML tags, avoiding standard LaTeX delimiter rendering clashes.
        """
        html = '<div style="display: flex; flex-wrap: wrap; gap: 35px; align-items: flex-start; padding: 5px 0;">'
        
        for title, item_content in items.items():
            if isinstance(item_content, tuple) and len(item_content) == 2:
                matrix_data, scalar_prefix = item_content
            else:
                matrix_data, scalar_prefix = item_content, ""
                
            rows = len(matrix_data)
            cols = len(matrix_data[0]) if rows > 0 else 0
            
            matrix_html = f'''
            <div style="text-align: center; margin-bottom: 15px;">
                <span style="font-weight: bold; color: #00a6d6; font-size: 14px; display: block; margin-bottom: 8px;">{title}</span>
                <div style="display: inline-flex; align-items: stretch; justify-content: center; vertical-align: middle;">
            '''
            
            if scalar_prefix:
                matrix_html += f'<div style="display: flex; align-items: center; font-family: \'Times New Roman\', Times, serif; font-size: 15px; font-weight: bold; margin-right: 6px; opacity: 0.9;">{scalar_prefix}</div>'
                
            # Using flex alignment combined with border lines makes sure the brackets scale dynamically with data lines
            matrix_html += f'''
                    <div style="border-left: 2px solid currentColor; border-top: 2px solid currentColor; border-bottom: 2px solid currentColor; width: 5px; margin-right: 5px; min-height: 24px;"></div>
                    
                    <div style="display: inline-grid; grid-template-columns: repeat({cols}, auto); gap: 10px 14px; align-items: center; justify-items: center; padding: 4px 6px; font-family: \'Times New Roman\', Times, serif; font-size: 15px; line-height: 1.2;">
            '''
            
            for r in range(rows):
                for c in range(cols):
                    val = matrix_data[r][c]
                    
                    if isinstance(val, str):
                        val_str = val
                    else:
                        evaluated_val = sp.sympify(val).evalf(decimal_places)
                        val_str = f"{float(evaluated_val):.{decimal_places}f}".rstrip('0').rstrip('.') if isinstance(evaluated_val, (sp.Float, float)) else str(evaluated_val)
                        if val_str in ["-0", "-0.0", ""]:
                            val_str = "0"
                            
                    matrix_html += f'<span>{val_str}</span>'
                    
            matrix_html += f'''
                    </div>
                    
                    <div style="border-right: 2px solid currentColor; border-top: 2px solid currentColor; border-bottom: 2px solid currentColor; width: 5px; margin-left: 5px; min-height: 24px;"></div>
                </div>
            </div>
            '''
            html += matrix_html
            
        html += '</div>'
        display(HTML(html))

def invert(matrix):

    m, n = np.array(matrix).shape
    if m != n:
        raise Exception('The matrix to be inverted must be squared!')
    
    inverse = inv(matrix)
    title = 'Inverted matrix'
    temp = project()
    temp.display_matrices({title: inverse}, 3)
    
    

