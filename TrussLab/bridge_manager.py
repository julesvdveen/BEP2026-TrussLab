import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from scipy.signal import find_peaks
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint

from .myDAQ_manager import myDAQ_sweep, logs_to_U

# ------------------------------------------------------------------------------------
#                                 DEFAULT SETTINGS:
# ------------------------------------------------------------------------------------

height = 160 # plate height in mm
width_0 = 100 # distance between node 0 and the first rigid node position (left) in mm
rigid_positions = [0, 100*(np.sqrt(2)-1), 100, 100*(2*np.sqrt(2)-1)] # mm

# ------------------------------------------------------------------------------------
                
class Bridge:
    """
    Bridge class for bridge structure analysis. Has two main functions: calculating the results and 
    visualising the results. 
    The calculation of the results are lists of the resistance values per beam.
    The visualising of the results are lists that contain the positions of the nodes.

    Attributes:

        library (dict): beam library applied for bridge construction.
        topology (list): topology list describing bridge structure.
        n (int): number of nodes/joints for given topology.
        k (int): number of (unique) beams/vertices in the structure.
        rigid (int): final node setting. If 0, horizontal DOF node. Otherwise, integers
                     [1, 2, 3] assign a rigid node position within the node slot.
        R_init (list): initial resistance vector 
        x_init (list): (estimated) undeformed node position vector [x0, y0, x1, ...].
        current_R (list): latest estimated resistance vector.
        current_x (list): latest estimated node position vector.

    Methods:
        construct: assigns topology to bridge object, following the order of the provided 
                   library. Also enables user to provide the state of the final node.
        solve_U: helper method to return voltage vector for all nodes n for given frequency, reference
                 resistance and beam resistance vector. 
        initial_response: returns the initial voltage response of the undeformed bridge,
                          with plotting option.
        initial_position: returns estimated node position vector for initial beam lengths.
        x_plot: helper method to plot bridge configuration for given node position vector.
        sweep: returns estimated resistance and node position vector for initialized bridge
               structure, using either myDAQ data or self-simulated (noisy) data.
    """

    def __init__(self):
        """
        Defining all the attributes as 'None', this way we get apprehensible errors.
        """
        self.library = None # beam library used for bridge construction
        self.topology = None # Topology list
        self.n = None # Number of nodes/joints
        self.k = None # Number of unique beams (in library).
        self.rigid = None # Final node rigidly constrained or single DOF

        self.R_init = None # Initial resistance vector
        self.x_init = None # Initial (estimated) node position vector

        self.current_R = None # Latest (estimated) resistance vector
        self.current_x = None # Latest (estimated) node position vector
    
    def construct(self, library:dict, topology:list, rigid:int=0):
        """
        Assigns topology to bridge object, following the order of the provided 
        library. Also enables user to provide the state of the final node.
        It also defines the R_init/R_0. And it uses the function initial_position() to
        define the x_init. 
        """
    
        self.__init__()

        k_lib = len(library)
        k_top = len(topology)
        if k_lib != k_top:
            raise Exception(f'topology list dimension {k_top} does not match library dimension {k_lib}')
        self.k = k_top

        self.library = library
        self.topology = topology
        self.rigid = rigid

        max_list = np.zeros(len(topology))
        for i in range(len(topology)):
            max_list[i] = max(topology[i])
        self.n = int(np.max(max_list)) + 1

        self.R_init = [beam.R_0 for beam in list(self.library.values())]
        self.x_init = self.initial_position()

        self.current_R = self.R_init
        self.current_x = self.x_init
        
    def solve_U(self, f_arr, R_ref, R_vec=None, U_in=1, noise=False, **kwargs):
        """
        helper method to return voltage vector for all nodes n for given frequency, reference
        resistance and beam resistance vector. 
        It uses the properties from the beam_manager to calculate the voltage vector with the Nodal Admittance Matrix.
        It is able to produce noise, which can be important to understand the sensitivity of the method in the theory.
        
        This function is the forward method: calculating the U_out using the known resistances.
        """
        w_arr = f_arr * 2*np.pi

        # First, the NAM is constructed for all frequencies in parallel:

        NAM = np.zeros((len(w_arr), self.n, self.n), dtype=complex) # shape (len(w_arr), n, n)
        beams = list(self.library.values())

        for k, beam in enumerate(beams):

            # Individual beam admittance:
            if R_vec is None:
                y_beam = beam.admittance(w_arr) # shape (len(w_arr))
            else:
                y_beam = beam.admittance(w_arr, R_vec[k])

            i, j = self.topology[k] # Individual beam position

            # Diagonal NAM elements:
            NAM[:,i,i] += y_beam
            NAM[:,j,j] += y_beam

            # Off-diagonal NAM elements:
            NAM[:,i,j] -= y_beam
            NAM[:,j,i] -= y_beam

        NAM[:,-1,-1] += 1/R_ref

        # Next, we solve for all U vectors in parallel:
        Y_RR = NAM[:,1:,1:]
        Y_RG = NAM[:,1:,0]

        U_R = np.linalg.solve(Y_RR, (-U_in*Y_RG)[..., np.newaxis]).squeeze(-1) # shape (len(w_arr), n-1)
        U_out = U_R[:,-1]

        if noise:

            abs_noise = kwargs.get('abs_noise', 0.005) # 5 mV
            sigma = kwargs.get('sigma', 0.005)
            rng = kwargs.get('rng')
            if rng is None: rng = np.random.default_rng()

            noise_scale = abs_noise + (sigma * np.abs(U_out))
            
            noise_real = rng.normal(0, noise_scale)
            noise_imag = rng.normal(0, noise_scale)
            
            U_out += noise_real + 1j * noise_imag

        return U_out

    def U_plot(self, f_arr, U):
        """
        Plotting the U_out
        """

        U_out_mag = np.abs(U)
        U_out_deg = np.degrees(np.angle(U))

        plt.figure(figsize=(10, 6))

        # Plot Magnitude
        plt.subplot(2, 1, 1)
        plt.semilogx(f_arr, U_out_mag, color='b')
        plt.ylabel('Gain')
        plt.grid(True, which="both")
        plt.title('Initial Bridge Frequency Response')

        # Plot Phase:
        plt.subplot(2, 1, 2)
        plt.semilogx(f_arr, U_out_deg, color='b')
        plt.ylabel('Phase (deg)')
        plt.xlabel('Frequency (Hz)')
        plt.grid(True, which="both", ls="-")
    
    def initial_response(self, f_arr, R_ref, plot=False):
        """
        This is being used by the notebooks, plotting the bode plot for known resistance values.
        """
        U_out = self.solve_U(f_arr, R_ref)

        if plot:
            self.U_plot(f_arr, U_out)
        
        return U_out

    def solve_x(self, R_vec=None, x_guess=None, plot=False):
        """
        Calculates the deformed node positions based on the deformation resistance
        vector. The rigid node is used as origin (0,0). mm and Ohms are used as units.

        Arguments:
            R_vec (list): deformed resistance values corresponding to the structure's beams.
            x_guess (list): initial guess for node position vector.
            plot (bool): option to provide plot of estimated bridge configuration.

        Returns:
            x (list): estimated node positions in shape [x0, y0, x1, ...].
        """

        if R_vec is None:
            # Undeformed bridge: R_init=R_0
            R_vec = self.R_init

        if len(R_vec) != self.k:
            raise Exception('R_vec dimension does not match the number of present beams.')

        # Calculate every beam length:
        l = np.zeros(self.k)
        for i, (beam, R) in enumerate(zip(list(self.library.values()), R_vec)):
            # alpha (mm/ohm) is experimentally obtained  
            l[i] = beam.l_0 + (R - beam.R_0) * beam.alpha

        x_n = width_0 + rigid_positions[self.rigid - 1] # Final node x-coordinate
        topo_arr = np.array(self.topology)

        # Define cost function:
        def cost(x_R):
            """
            Saving 
            """

            # Construct full position vector:
            x_0 = np.zeros(2) # Assign coordinate system origin to 0-th node.
            if self.rigid:
                x = np.concatenate((x_0, x_R, [x_n], [0]))
            else:
                x = np.concatenate((x_0, x_R, [0]))

            # Vectorize length calculation:
            X = x.reshape(-1, 2) # reshape position vector into coordinate matrix, shape (n, 2)

            xi = X[topo_arr[:,0],:] # shape (k, 2)
            xj = X[topo_arr[:,1],:] # shape (k, 2)

            dx = xi - xj # shape (k, 2)
            l_est = np.sqrt(np.sum(dx**2, axis=1)) # shape (k)
            gem_err_individual_squared = (l_est - l)**2

            return np.sum(gem_err_individual_squared)
        
        # Construct initial guess:
        if x_guess is None:
            if self.rigid:
                x_guess = self.current_x[2:-2]
            else:
                x_guess = self.current_x[2:-1] # x_(n-1) must also be found!
        else:
            if self.rigid:
                x_guess = x_guess[2:-2]
            else:
                x_guess = x_guess[2:-1] # x_(n-1) must also be found!

        # Initiate bounds:
        bounds_free = [(0, width_0 + rigid_positions[-1] + 30), (0, height)] * (self.n - 2)
        if self.rigid:
            bounds = bounds_free
        else:
            bound_xn = (width_0, width_0 + rigid_positions[-1] + 30) # Full slot length (including 30 mm deformation space)
            bounds = bounds_free + [bound_xn]
                
        res = minimize(cost, x0=x_guess,  method='SLSQP', bounds=bounds, options={'ftol': 1e-8})
        x_R = res.x

        x_0 = np.zeros(2) # Assign coordinate system origin to 0-th node.
        if self.rigid:
            x = np.concatenate((x_0, x_R, [x_n], [0]))
        else:
            x = np.concatenate((x_0, x_R, [0]))

        if plot:
            self.x_plot(x)
        
        return x

    def initial_position(self, plot=False, min_distance=60.0):
        """
        Calculates the deformed node positions based on the deformation resistance
        vector. Also returns the geometric error, as an constrained for the predict_R
        method. The rigid node is used as origin (0,0). mm and Ohms are used as units.

        Returns:
            x (list): estimated node positions in shape [x0, y0, x1, ...].
        """

        l = [beam.l_0 for beam in list(self.library.values())]

        x_n = width_0 + rigid_positions[self.rigid - 1] # Far right rigid slot position
        topo_arr = np.array(self.topology)

        # Define cost function:
        def cost(x_R):
            # Construct full position vector:
            x_0 = np.zeros(2) # Assign coordinate system origin to 0-th node.
            if self.rigid:
                x = np.concatenate((x_0, x_R, [x_n], [0]))
            else:
                x = np.concatenate((x_0, x_R, [0]))

            # Vectorize length calculation:
            X = x.reshape(-1, 2) # reshape position vector into coordinate matrix, shape (n, 2)

            xi = X[topo_arr[:,0],:] # shape (k, 2)
            xj = X[topo_arr[:,1],:] # shape (k, 2)

            dx = xi - xj # shape (k, 2)
            l_est = np.sqrt(np.sum(dx**2, axis=1)) # shape (k)
            gem_err_individual_squared = (l_est - l)**2

            return np.sum(gem_err_individual_squared)
        
        def constraint_min_distance(x_R):
            x_0 = np.zeros(2)
            if self.rigid:
                x = np.concatenate((x_0, x_R, [x_n], [0]))
            else:
                x = np.concatenate((x_0, x_R, [0]))
            
            X = x.reshape(-1, 2)
            distances = []
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    dist = np.linalg.norm(X[i] - X[j])
                    distances.append(dist)
            
            return np.array(distances)

        # Initiate bounds:
        bounds_free = [(0, width_0 + rigid_positions[-1] + 30), (0, height)] * (self.n - 2)
        if self.rigid:
            bounds = bounds_free
        else:
            bound_xn = (width_0, width_0 + rigid_positions[-1] + 30) # Full slot length (including 30 mm deformation space)
            bounds = bounds_free + [bound_xn]

        num_node_pairs = len(list(range(self.n))) * (len(list(range(self.n))) - 1) // 2
        constraint = NonlinearConstraint(
            constraint_min_distance,
            lb=min_distance * np.ones(num_node_pairs),
            ub=np.inf * np.ones(num_node_pairs)
            )

        res = differential_evolution(cost, bounds=bounds, constraints=constraint, 
                                  strategy='best1bin',
                                  maxiter=1000,
                                  popsize=15,
                                  tol=1e-8,
                                  seed=None,
                                  polish=True)

        x_R = res.x
            
        x_0 = np.zeros(2) # Assign coordinate system origin to 0-th node.
        if self.rigid:
            x = np.concatenate((x_0, x_R, [x_n], [0]))
        else:
            x = np.concatenate((x_0, x_R, [0]))

        if plot:
            self.x_plot(x)

        return x

    def x_plot(self, x=None, anim=False):
    
        if x is None:
            x = self.current_x
        
        beam_ids = [beam.ID for beam in list(self.library.values())]
        topology = self.topology
        rigid = bool(self.rigid)
        coords = x.reshape(-1,2)

        fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(12, 6),
                                gridspec_kw={"width_ratios": [4, 1.5]},
                                constrained_layout=False)
    
        def plot_rigid_node(ax, coords, node_id, width=10, height=8):
            artists = []
            x, y = coords[node_id]

            triangle = Polygon([[x, y], [x - width / 2, y - height], [x + width / 2, y - height]], closed=True, facecolor="#82b1fe", edgecolor="black", linewidth=2.5, zorder=0)
            ax.add_patch(triangle)
            artists.append(triangle)

            line, = ax.plot([x - width * 0.7, x + width * 0.7], [y - height, y - height], color="black", linewidth=2.5, zorder=0)
            artists.append(line)

            n_hatches = 5
            spacing = width * 1.4 / (n_hatches + 1)

            for i in range(n_hatches):
                x0 = x - width * 0.7 + (i + 1) * spacing
                hatch, = ax.plot([x0 - 3, x0 + 3], [y - height - 4, y - height], color="black", linewidth=2, zorder=0)
                artists.append(hatch)

            return artists


        def plot_sliding_support(ax, coords, node_id, width=10, height=8):
            artists = []
            x, y = coords[node_id]

            triangle = Polygon([[x, y], [x - width / 2, y - height + 3.5], [x + width / 2, y - height + 3.5]], closed=True, facecolor="#82b1fe", edgecolor="black", linewidth=2.5, zorder=0)
            ax.add_patch(triangle)
            artists.append(triangle)

            ground_y = y - height

            line, = ax.plot([x - width * 0.7, x + width * 0.7], [ground_y, ground_y], color="black", linewidth=2.5, zorder=0)
            artists.append(line)

            n_hatches = 5
            spacing = width * 1.4 / (n_hatches + 1)

            for i in range(n_hatches):
                x0 = x - width * 0.7 + (i + 1) * spacing
                hatch, = ax.plot([x0 - 3, x0 + 3], [ground_y - 4, ground_y], color="black", linewidth=2, zorder=0)
                artists.append(hatch)

            return artists

        # Beams
        for i, j in topology:
            xi, yi = coords[i]; xj, yj = coords[j]
            ax.plot([xi, xj], [yi, yj], color="black", linewidth=7.8,
                    solid_capstyle="round", zorder=1)

        # Supports
        plot_rigid_node(ax, coords, node_id=0, width=18, height=24)
        if rigid == True:
            plot_rigid_node(ax, coords, node_id=-1, width=18, height=24)
        elif rigid == False:
            plot_sliding_support(ax, coords, node_id=-1, width=18, height=24)
        else:
            raise ValueError("rigid moet True of False zijn.")
            
        # Nodes
        ax.scatter(coords[:, 0], coords[:, 1], color="#fed030",
                edgecolors="black", linewidths=2.5, s=100, zorder=3)

        # Beam-namen / nummers, gedraaid met de balk mee
        for k, (i, j) in enumerate(topology):
            xi, yi = coords[i]
            xj, yj = coords[j]

            x_mid = (xi + xj) / 2
            y_mid = (yi + yj) / 2

            dx = xj - xi
            dy = yj - yi

            angle = np.degrees(np.arctan2(dy, dx))

            # Voorkom tekst ondersteboven
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            ax.text(
                x_mid, y_mid, beam_ids[k],
                color="#fed030",
                weight="bold",
                fontsize=7,
                ha="center",
                va="center",
                rotation=angle,
                rotation_mode="anchor",
                zorder=5
            )

        # Node-nummers
        for idx, (x_node, y_node) in enumerate(coords):
            ax.text(x_node, y_node-0.5, str(idx),
                    color="black", weight="bold", fontsize=8,
                    ha="center", va="center", zorder=6)

        # Tabel
        # Tabel-as exact uitlijnen met grafiek-as
        fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.16, wspace=0.06)

        pos_ax = ax.get_position()
        pos_table = ax_table.get_position()

        ax_table.set_position([
            pos_table.x0,
            pos_ax.y0,
            pos_table.width,
            pos_ax.height
        ])
        
        ax_table.axis("off")

        node_data = []
        for idx, (x_node, y_node) in enumerate(coords):
            node_data.append([f"{idx}", f"{x_node:.1f}", f"{y_node:.1f}"])

        beam_data = []
        for k, (i, j) in enumerate(topology):
            xi, yi = coords[i]
            xj, yj = coords[j]
            length = np.sqrt((xj - xi)**2 + (yj - yi)**2)
            beam_data.append([beam_ids[k], f"{i}-{j}", f"{length:.1f}"])

        # Node table
        node_table = ax_table.table(
            cellText=node_data,
            colLabels=["Node", "x [mm]", "y [mm]"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, 0.6, 1.0, 0.362]
        )

        # Beam table
        beam_table = ax_table.table(
            cellText=beam_data,
            colLabels=["Beam", "Nodes", "L [mm]"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, 0.038, 1.0, 0.53]
        )
        
        
        for table in [node_table, beam_table]:
            table.auto_set_font_size(False)
            table.set_fontsize(8)

            for (row, col), cell in table.get_celld().items():
                cell.set_linewidth(0.5)

                if row == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#82b1fe")

        # Styling
        fig.patch.set_facecolor("white")
        ax.set_facecolor('#f0f2f5')
        ax_table.set_facecolor("white")

        ax.set_title("Truss Geometry Layout", weight="bold", fontsize=12)
        ax.set_xlabel("X-coordinate (mm)", fontsize=9)
        ax.set_ylabel("Y-coordinate (mm)", fontsize=9)

        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        ax.set_xlim(-20, 345) 
        ax.set_ylim(-30, 160)
        ax.set_yticks([0, 25, 50, 75, 100, 125, 150])

        def animate_truss(x_final, scale=1.0, n_frames=60, interval=30):

            x_init = np.asarray(self.x_init, dtype=float) # Jules dit was jij!!!!
            x_final = np.asarray(x_final, dtype=float)

            coords_init = x_init.reshape(-1, 2)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.set_facecolor('#f0f2f5')

            for i, j in topology:
                ax.plot([coords_init[i, 0], coords_init[j, 0]], [coords_init[i, 1], coords_init[j, 1]], color="lightgray", linewidth=7.8, linestyle="--", solid_capstyle="round", zorder=1)

            beam_lines = [(ax.plot([], [], color="black", linewidth=7.8, solid_capstyle="round", zorder=2)[0], i, j) for i, j in topology]

            nodes = ax.scatter([], [], color="#fed030", edgecolors="black", linewidths=2.5, s=100, zorder=3)

            node_labels = [ax.text(0, 0, str(i), color="black", weight="bold", fontsize=8, ha="center", va="center", zorder=6) for i in range(coords_init.shape[0])]

            beam_labels = [ax.text(0, 0, beam_ids[k], color="#fed030", weight="bold", fontsize=7, ha="center", va="center", rotation_mode="anchor", zorder=5) for k in range(len(topology))]

            ax.set_xlim(-20, 345)
            ax.set_ylim(-30, 160)
            ax.set_yticks([0, 25, 50, 75, 100, 125, 150])

            ax.set_aspect("equal")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_title("Truss Deformation Animation", weight="bold", fontsize=12)
            ax.set_xlabel("X-coordinate (mm)", fontsize=9)
            ax.set_ylabel("Y-coordinate (mm)", fontsize=9)

            support_artists = []

            def update(frame):
                nonlocal support_artists

                alpha = frame / (n_frames - 1)
                coords_now = (x_init + alpha * scale * (x_final - x_init)).reshape(-1, 2)

                for artist in support_artists:
                    artist.remove()

                support_artists = []
                support_artists += plot_rigid_node(ax, coords_now, node_id=0, width=18, height=24)

                if rigid:
                    support_artists += plot_rigid_node(ax, coords_now, node_id=-1, width=18, height=24)
                else:
                    support_artists += plot_sliding_support(ax, coords_now, node_id=-1, width=18, height=24)

                for line, i, j in beam_lines:
                    line.set_data([coords_now[i, 0], coords_now[j, 0]], [coords_now[i, 1], coords_now[j, 1]])

                nodes.set_offsets(coords_now)

                for i, txt in enumerate(node_labels):
                    txt.set_position((coords_now[i, 0], coords_now[i, 1] - 0.5))

                for k, (i, j) in enumerate(topology):
                    xi, yi = coords_now[i]
                    xj, yj = coords_now[j]

                    angle = np.degrees(np.arctan2(yj - yi, xj - xi))

                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180

                    beam_labels[k].set_position(((xi + xj) / 2, (yi + yj) / 2))
                    beam_labels[k].set_rotation(angle)

                return [line for line, _, _ in beam_lines] + [nodes] + node_labels + beam_labels + support_artists

            ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False, repeat=True)
            plt.close(fig)

            return ani

        if anim:
            return animate_truss(x)

    def sweep(self, R_ref, f_logrange=(3, np.log10(2e4)), freqsteps=1000, myDAQ=False, myDAQlog=None, resonance_sweep=False, geo_constraint=False, 
                  max_iter=500, position_plot=False, error_plot=False, response_comparison_plot=False, **kwargs):
        
        # ----------------------------------------------------------------------
        # STEP 0: INITIALISATION
        # ----------------------------------------------------------------------
        
        f_start, f_stop = f_logrange
        f_arr = np.logspace(f_start, f_stop, freqsteps)
        
        f_base_targets = None 
        f_diff_targets = None

        abs_noise = None
        sigma = None
        rng = None

        if not myDAQ and myDAQlog:
            raise Exception('if a myDAQ log should be provided, the myDAQ argument should be TRUE.')
        
        if not myDAQ:
            R_def = kwargs.get('R_def', None) 

        noise = kwargs.get('noise', False)

        if not myDAQ and noise:
            abs_noise = kwargs.get('abs_noise', 0.005) # 5 mV
            sigma = kwargs.get('sigma', 0.005) # 0.5% of signal strength
            rng = kwargs.get('rng')
            if rng is None: rng = np.random.default_rng()

        # ----------------------------------------------------------------------
        # STEP 1: PERFORM FREQUENCY SWEEP
        # ----------------------------------------------------------------------

        if myDAQlog is None:

            if resonance_sweep:
                
                f_arr_base = np.logspace(f_start, f_stop, 15000)
                rel_df = kwargs.get('rel_df', 0.05) # Default to +/- 5% of the center frequency

                # Base:
                U_out_base = self.initial_response(f_arr_base, R_ref)
                mag_base = np.abs(U_out_base)
                base_peaks, _ = find_peaks(mag_base, prominence=0.01)
                base_dips, _  = find_peaks(-mag_base, prominence=0.01)
                base_indices  = np.unique(np.concatenate((base_peaks, base_dips)))
                f_base_targets = f_arr_base[base_indices]
                print(f"Targeting {len(f_base_targets)} base features at (Hz): {np.round(f_base_targets, 1)}")

                # Differential:
                R_nominal = self.R_init
                beam_keys = list(self.library.keys())

                all_beam_candidates = {}  # Stores sorted peak choices for each beam
                dU_history = {}           # Remembers each beam's individual derivative array
                dR = 10                   # 10 Ohm central difference perturbation step

                for k in range(self.k):
                    # Upper perturbation
                    R_plus = R_nominal.copy()
                    R_plus[k] += dR
                    U_plus = self.solve_U(f_arr_base, R_ref, R_vec=R_plus, noise=False)

                    # Lower perturbation
                    R_minus = R_nominal.copy()
                    R_minus[k] -= dR
                    U_minus = self.solve_U(f_arr_base, R_ref, R_vec=R_minus, noise=False)

                    # Central Difference Derivative magnitude (Complex subtraction catches phase flips)
                    dU_k = np.abs(U_plus - U_minus) / (2 * dR) 
                    dU_history[k] = dU_k

                    peaks_k, _ = find_peaks(dU_k, prominence=1e-6)
                    
                    if len(peaks_k) == 0:
                        # Fallback to absolute maximum if no clean peaks stand out
                        peaks_k = np.array([np.argmax(dU_k)])
                        peak_heights = np.array([dU_k.max()])
                    else:
                        peak_heights = dU_k[peaks_k] 

                    # Sort candidates from HIGHEST sensitivity derivative to LOWEST
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    all_beam_candidates[k] = f_arr_base[peaks_k[sorted_indices]]

                    # Plots for checking sensitivity:
                    # fig, ax = plt.subplots()
                    # ax.plot(f_arr_base, dU_k, label='dU/dR')
                    # for candidate in all_beam_candidates[k]:
                    #     ax.axvline(candidate, linestyle = '--')
                    # ax.legend()
                    # ax.grid(True, alpha=.7)

                dU_min_floor = 50e-6
                allocated_frequencies_dict = {}
                f_diff_targets = []

                # Sorting:
                allocation_order = list(range(self.k))
                allocation_order.sort(key=lambda b: len(all_beam_candidates[b]))

                for k in allocation_order:
                    options = all_beam_candidates[k]
                    assigned = False
                    my_dU_k = dU_history[k] 
                    
                    # Track why assignment failed for diagnostics
                    failed_due_to_floor = False
                    failed_due_to_conflict = False
                    
                    for f_opt in options:
                        idx_f = np.where(f_arr_base == f_opt)[0][0]
                        
                        # Verify candidate stands out safely above the sensor noise floor
                        if my_dU_k[idx_f] < dU_min_floor:
                            failed_due_to_floor = True
                            continue

                        # Conflict check: Ensure option isn't too close to an already allocated band
                        if not any(np.isclose(f_opt, f_alloc, rtol=rel_df) for f_alloc in f_diff_targets):
                            f_diff_targets.append(f_opt)
                            allocated_frequencies_dict[k] = f_opt
                            print(f"  R_{beam_keys[k]}: Unique feature allocated at {f_opt:.1f} Hz")
                            assigned = True
                            break
                        else:
                            failed_due_to_conflict = True
                    
                    # Fallback: If forced into an overlap, accept the absolute best option anyway
                    if not assigned:
                        if failed_due_to_floor and not failed_due_to_conflict:
                            message = 'no sensitivity dU/dR above 100 uV/Ohm was found'
                        elif failed_due_to_conflict and not failed_due_to_floor:
                            message = 'no unique frequency (after selecting the frequencies above) was found'
                        else:
                            message = 'available unique options fell below the sensitivity floor'
                            
                        fallback_f = options[0]
                        f_diff_targets.append(fallback_f)
                        allocated_frequencies_dict[k] = fallback_f
                        print(f"R_{beam_keys[k]}: {message}. Utilizing peak at {fallback_f:.1f} Hz")

                # Map assigned target frequencies back to indices on the master grid
                f_diff_targets = np.array(f_diff_targets)
                diff_indices = np.array([np.where(f_arr_base == allocated_frequencies_dict[k])[0][0] for k in range(self.k)])
                
                # Combine base system features and decoupled individual unique beam features safely
                all_indices = np.unique(np.concatenate((base_indices, diff_indices)))
                f_sweep = f_arr_base[all_indices]

                # Merge closeby bands:
                raw_ranges = []
                for f in f_sweep:
                    relative_df = f * rel_df
                    f_lower = max(f - relative_df, 1e-3)
                    f_upper = f + relative_df
                    raw_ranges.append([f_lower, f_upper])
                
                # Cleanly sort and merge overlapping windows globally
                raw_ranges.sort(key=lambda x: x[0])
                merged_ranges = []
                if raw_ranges:
                    curr_start, curr_stop = raw_ranges[0]
                    for next_start, next_stop in raw_ranges[1:]:
                        if next_start <= curr_stop:
                            curr_stop = max(curr_stop, next_stop)
                        else:
                            merged_ranges.append([curr_start, curr_stop])
                            curr_start, curr_stop = next_start, next_stop
                    merged_ranges.append([curr_start, curr_stop])

                # Calculate uniform steps per decade based on merged windows
                log_widths = np.array([np.log10(stop/start) for start, stop in merged_ranges])
                total_log_width = np.sum(log_widths) if np.sum(log_widths) > 0 else 1.0
                steps_per_decade = int(np.round(freqsteps / total_log_width))

                # Reconstruct clean, individual segments for the plotting loops
                plot_segments = []
                for start, stop in merged_ranges:
                    # Distribute total steps proportionally across logarithmic width
                    frac = np.log10(stop/start) / total_log_width
                    steps_seg = max(int(np.round(freqsteps * frac)), 10)
                    plot_segments.append(np.logspace(np.log10(start), np.log10(stop), steps_seg))

                if myDAQ:
                    f_start_list = [r[0] for r in merged_ranges]
                    f_stop_list = [r[1] for r in merged_ranges]

                    save_filename = kwargs.get('save_filename', None)
                    if save_filename is None:
                        raise Exception('Please specify a saving filename for the frequency sweep log (save_filename = ... ).')
                    
                    f_arr, U_out_mes = myDAQ_sweep(f_start_list, f_stop_list, [steps_per_decade]*len(f_start_list), save_filename)
                
                else:
                    # Concatenate your matching segments linearly to maintain sorted order
                    f_arr = np.concatenate(plot_segments) if plot_segments else np.array([])
                    
                    # Gather measurements from internal numerical simulation
                    U_out_mes = self.solve_U(f_arr, R_ref, R_vec=R_def, noise=noise, abs_noise=abs_noise, sigma=sigma, rng=rng)
                    
            # If resonance sweep is turned off:
            else:   
                if myDAQ:
                    save_filename = kwargs.get('save_filename', None)
                    if save_filename is None:
                        raise Exception('Please insert a saving filename for the frequency sweep log (save_filename = ... ).')
                    steps_per_decade = int(np.round(freqsteps / (f_stop - f_start)))
                    f_arr, U_out_mes = myDAQ_sweep([10**f_start], [10**f_stop], [steps_per_decade], save_filename)
                else:
                    f_arr = np.logspace(f_start, f_stop, freqsteps)
                    U_out_mes = self.solve_U(f_arr, R_ref, R_vec=R_def, noise=noise,
                                            abs_noise=abs_noise, sigma=sigma, rng=rng)
        else:
            f_arr, U_out_mes = logs_to_U(myDAQlog)


        # ----------------------------------------------------------------------
        # STEP 2: USE OPTIMALISATION ALGORITHM TO OBTAIN R AND x VECTOR
        # ----------------------------------------------------------------------

        def cost(param):

            R = param[:self.k]
            x_R = param[self.k:]

            U_out_pred = self.solve_U(f_arr, R_ref, R_vec=R)
            U_mag_err = np.abs(U_out_mes - U_out_pred)
            J = np.sqrt(np.mean(U_mag_err**2))

            if not geo_constraint: return J
            else:

                # Calculate every beam length:
                l = np.zeros(self.k)
                for i, (beam, Ri) in enumerate(zip(list(self.library.values()), R)):
                    l[i] = beam.l_0 + (Ri - beam.R_0) * beam.alpha

                topo_arr = np.array(self.topology)
                x_n = width_0 + rigid_positions[self.rigid - 1]

                # Construct full position vector:
                x_0 = np.zeros(2) # Assign coordinate system origin to 0-th node.
                if self.rigid:
                    x = np.concatenate((x_0, x_R, [x_n], [0]))
                else:
                    x = np.concatenate((x_0, x_R, [0]))

                # Vectorize length calculation:
                X = x.reshape(-1, 2) # reshape position vector into coordinate matrix, shape (n, 2)

                xi = X[topo_arr[:,0],:] # shape (k, 2)
                xj = X[topo_arr[:,1],:] # shape (k, 2)

                # Calculate geometric error:
                dx = xi - xj # shape (k, 2)
                l_est = np.sqrt(np.sum(dx**2, axis=1)) # shape (k)
                gem_err_individual_squared = (l_est - l)**2 # shape (k)
                J_geo = np.sum(gem_err_individual_squared)

                lam = kwargs.get('lam', 5e-4)
                J += lam * J_geo
                return J

        bounds = [(1.0, 1e3)] * self.k
        if geo_constraint:

            x_n = width_0 + rigid_positions[self.rigid - 1]
            bounds_geo_free = [(0, width_0 + rigid_positions[-1] + 30), (0, height)] * (self.n - 2)

            if self.rigid:
                bounds_geo = bounds_geo_free
            else:
                bound_geo_xn = (width_0, width_0 + rigid_positions[-1] + 30) # Full slot length
                bounds_geo = bounds_geo_free + [bound_geo_xn]
            
            bounds += bounds_geo

        res_global = differential_evolution(cost, bounds, maxiter=max_iter, tol=1e-8, seed=42)
        if not res_global.success:
            raise Exception(f'The global optimization (differential evolution) did not converge: {res_global.message}')
        
        res = minimize(cost, res_global.x, method='SLSQP', bounds=bounds, options={'ftol': 1e-8})
        if not res.success:
            raise Exception(f'The local optimization (SLSQP) did not converge: {res.message}')
        
        R_pred = res.x[:self.k]
        if geo_constraint:
            x_R_pred = res.x[self.k:]
            x_0 = np.zeros(2) # Assign coordinate system origin to 0-th node.
            if self.rigid:
                x_pred = np.concatenate((x_0, x_R_pred, [x_n], [0]))
            else:
                x_pred = np.concatenate((x_0, x_R_pred, [0]))

        # ----------------------------------------------------------------------
        # STEP 3 (OPTIONAL): PLOTTING
        # ----------------------------------------------------------------------

        if position_plot:
            if not geo_constraint:
                print('No position plot available, since the geometric constraint was toggled off. Use x_plot(R_vec = R_pred).')
            else: self.x_plot(x_pred)

        if error_plot:
            
            R_var = kwargs.get('R_var', [0])
            R_def = kwargs.get('R_def', None)

            Rdev = 200.0
            steps = 50

            if len(R_var) > 2:
                raise Exception('The error can be displayed for up to two resistance variables at max.')
            
            print('R-values found, generating plot...')
            
            if len(R_var) == 1:

                idx = R_var[0]

                if R_def == None:
                    r_space = np.linspace(R_pred[idx] - Rdev, R_pred[idx] + Rdev, steps)
                else:
                    r_space = np.linspace(np.min([R_def[idx], R_pred[idx]]) - Rdev,
                                          np.max([R_def[idx], R_pred[idx]]) + Rdev,
                                          steps)
                
                R_temp = np.full((self.k, steps), R_pred[:, np.newaxis])
                R_temp[idx,:] = r_space # shape (k, steps)

                if geo_constraint:
                    if self.rigid:
                        costs = [cost(np.concatenate((R, x_pred[2:-2]))) for R in R_temp.T]
                    else:
                        costs = [cost(np.concatenate((R, x_pred[2:-1]))) for R in R_temp.T]
                
                else: costs = [cost(R_temp[:,i]) for i in range(steps)]

                plt.figure(figsize=(8, 5))
                plt.plot(r_space, costs, label=f'Cost $J$')
                plt.axvline(R_pred[idx], color='purple', linestyle='--', label=f'Predicted $R_{{{list(self.library.keys())[idx]}}}$')
                if R_def is not None:
                    plt.axvline(R_def[idx], color='g', linestyle='--', label=f'Correct $R_{{{list(self.library.keys())[idx]}}}$')
                plt.xlabel(f'Resistance $R_{{{list(self.library.keys())[idx]}}}$')
                plt.ylabel('Cost $J$ [V]')
                plt.title('RMSE Cost')
                plt.legend()
                plt.grid(True)

            if len(R_var) == 2:
                idx = R_var[0]
                idy = R_var[1]

                if R_def is None:
                    rx_space = np.linspace(R_pred[idx] - Rdev, R_pred[idx] + Rdev, steps)
                    ry_space = np.linspace(R_pred[idy] - Rdev, R_pred[idy] + Rdev, steps)
                else:
                    rx_space = np.linspace(np.min([R_def[idx], R_pred[idx]]) - Rdev,
                                          np.max([R_def[idx], R_pred[idx]]) + Rdev,
                                          steps)
                    ry_space = np.linspace(np.min([R_def[idy], R_pred[idy]]) - Rdev,
                                          np.max([R_def[idy], R_pred[idy]]) + Rdev,
                                          steps)

                X, Y = np.meshgrid(rx_space, ry_space)

                X_flat = X.ravel()
                Y_flat = Y.ravel()

                R_temp = np.full((X_flat.size, self.k), R_pred) # shape (steps**2, k)
                R_temp[:, idx] = X_flat
                R_temp[:, idy] = Y_flat

                if geo_constraint:
                    if self.rigid:
                        costs_flat = [cost(np.concatenate((R, x_pred[2:-2]))) for R in R_temp]
                    else:
                        costs_flat = [cost(np.concatenate((R, x_pred[2:-1]))) for R in R_temp]
                else: costs_flat = [cost(R_temp[i,:]) for i in range(X_flat.size)]

                costs = np.array(costs_flat).reshape(X.shape)
                c_min = costs.min()
                Jmax_capped = np.percentile(costs, 40) 
                if Jmax_capped <= c_min:
                    Jmax_capped = c_min + 1e-3 

                fig, ax = plt.subplots(figsize=(10, 7))
                levels = np.linspace(c_min, Jmax_capped, 30)
                cntr = ax.contourf(X, Y, costs, levels=levels, cmap='viridis', extend='max')
                line_cntr = ax.contour(X, Y, costs, levels=levels, colors='white', 
                                    linewidths=0.5, alpha=0.3)

                ax.plot(R_pred[idx], R_pred[idy], 'ro', markersize=8, label='Predicted $R$', zorder=5)
                if R_def is not None:
                    ax.plot(R_def[idx], R_def[idy], 'go', markersize=8, label='Correct $R$', zorder=5)

                ax.set_xlabel(f'$R_{{{list(self.library.keys())[idx]}}}$ ($\Omega$)')
                ax.set_ylabel(f'$R_{{{list(self.library.keys())[idy]}}}$ ($\Omega$)')
                
                plt.title('RMSE Cost')
                plt.legend(loc='upper right')
                
                colorbar_label = 'Cost $J$'
                if not geo_constraint:
                    colorbar_label += ' (V)'
                fig.colorbar(cntr, ax=ax, label=colorbar_label)

        if response_comparison_plot:

            if myDAQlog:
                log_f_diff = np.diff(np.log10(f_arr))
                if np.max(log_f_diff) > 2.0 * np.median(log_f_diff):
                    resonance_sweep = True

            R_def = kwargs.get('R_def', None)
            include_mes = kwargs.get('include_mes', True)

            if resonance_sweep and not include_mes:
                f_arr = np.logspace(f_start, f_stop, freqsteps)

            if not resonance_sweep or resonance_sweep and not include_mes:

                U_out_base = self.solve_U(f_arr, R_ref, R_vec=None)
                U_out_base_mag = np.abs(U_out_base) # Gain
                U_out_base_deg = np.degrees(np.angle(U_out_base))

                U_out_pred = self.solve_U(f_arr, R_ref, R_vec=R_pred)
                U_out_pred_mag = np.abs(U_out_pred) # Gain
                U_out_pred_deg = np.degrees(np.angle(U_out_pred))

                U_out_theory = self.solve_U(f_arr, R_ref, R_vec=R_def)
                U_out_theory_mag = np.abs(U_out_theory) # Gain
                U_out_theory_deg = np.degrees(np.angle(U_out_theory))

                dU_pred_mag =  U_out_pred_mag - U_out_base_mag # Gain Difference
                dU_pred_deg =  U_out_pred_deg - U_out_base_deg # Phase Difference

                dU_theory_mag = U_out_theory_mag - U_out_base_mag # Gain Difference
                dU_theory_deg = U_out_theory_deg - U_out_base_deg # Phase Difference

                fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

                axs[0].semilogx(f_arr, np.full(f_arr.size, 0), color='b', linestyle='--', alpha=0.5, label='Base')
                axs[0].semilogx(f_arr, dU_pred_mag, color='r', label='Predicted')
                axs[0].semilogx(f_arr, dU_theory_mag, color='y', linestyle='--', label='Theory')

                axs[1].semilogx(f_arr, np.full(f_arr.size, 0), color='b', linestyle='--', alpha=0.5, label='Base')
                axs[1].semilogx(f_arr, dU_pred_deg, color='r', label='Predicted')
                axs[1].semilogx(f_arr, dU_theory_deg, color='y', linestyle='--', label='Theory')

                if include_mes:

                    U_out_mes_mag = np.abs(U_out_mes) # Gain
                    U_out_mes_deg = np.degrees(np.angle(U_out_mes))

                    dU_mes_mag = U_out_mes_mag - U_out_base_mag # Gain Difference
                    dU_mes_deg = U_out_mes_deg - U_out_base_deg # Phase Difference
                    
                    axs[0].semilogx(f_arr, dU_mes_mag, color='g', label='Measured')
                    axs[1].semilogx(f_arr, dU_mes_deg, color='g', label='Measured')

                axs[0].set_ylabel(r'$\Delta$ Gain')
                axs[0].legend(loc='upper left')
                axs[0].grid(True, which="both", ls="-")

                axs[1].set_xlabel('Frequency (Hz)')
                axs[1].set_ylabel(r'$\Delta$ Phase (deg)')
                axs[1].grid(True, which="both", ls="-")

                # --- distinct plotting ---
                if f_base_targets is not None:
                    for ax in axs:
                        for f in f_base_targets:
                            ax.axvline(f, color='purple', linestyle=':', alpha=0.7)
                if f_diff_targets is not None:
                    for ax in axs:
                        for f in f_diff_targets:
                            ax.axvline(f, color='gold', linestyle=':', alpha=0.9, linewidth=1.5)

                plt.tight_layout()
        
            else:
                
                if 'plot_segments' in locals():
                    f_segments = plot_segments
                else:
                    log_f = np.log10(f_arr)
                    diffs = np.diff(log_f)
                    # Gaps between distinct tracked bands show prominent decade jumps (> 0.03 decades)
                    gap_indices = np.where(diffs > 0.03)[0]
                    f_segments = np.split(f_arr, gap_indices + 1)

                M = len(f_segments)
                
                fig, axs = plt.subplots(2, M, figsize=(3 * M + 2.5, 6), 
                                        sharey=False, sharex='col', squeeze=False, constrained_layout=True)
                
                start_idx = 0
                for col, f_seg in enumerate(f_segments):
                    end_idx = start_idx + len(f_seg)
                    
                    f_mes_seg = f_arr[start_idx:end_idx] 
                    U_mes = U_out_mes[start_idx:end_idx]
                
                    U_base = self.solve_U(f_mes_seg, R_ref, R_vec=None)
                    U_pred = self.solve_U(f_mes_seg, R_ref, R_vec=R_pred)
                    U_theo = self.solve_U(f_mes_seg, R_ref, R_vec=R_def)
                    
                    dG_pred = np.abs(U_pred) - np.abs(U_base)
                    dG_theo = np.abs(U_theo) - np.abs(U_base)
                    dG_mes  = np.abs(U_mes) - np.abs(U_base)
                    
                    dP_pred = np.degrees(np.angle(U_pred)) - np.degrees(np.angle(U_base))
                    dP_theo = np.degrees(np.angle(U_theo)) - np.degrees(np.angle(U_base))
                    dP_mes  = np.degrees(np.angle(U_mes)) - np.degrees(np.angle(U_base))
                    
                    ax_gain = axs[0, col]
                    
                    # FIRST: Plot messy measurements so they lie at the bottom layer
                    ax_gain.semilogx(f_mes_seg, dG_mes, color='g', label='Measured')
                    
                    # THEN: Overlay baseline, predictions, and theory on top
                    ax_gain.semilogx(f_mes_seg, np.zeros(f_mes_seg.size), 'b--', alpha=0.5)
                    ax_gain.semilogx(f_mes_seg, dG_pred, 'r-', label='Predicted')
                    ax_gain.semilogx(f_mes_seg, dG_theo, 'y--', label='Theory')
                    
                    ax_gain.grid(True, which="both", ls="-")
                    ax_gain.set_title(f"Band {col+1}", fontsize=11, weight='bold')
                    
                    # ---- Bottom Row: Delta Phase ----
                    ax_phase = axs[1, col]
                    
                    # FIRST: Plot messy measurements so they lie at the bottom layer
                    ax_phase.semilogx(f_mes_seg, dP_mes, color='g')
                    
                    # THEN: Overlay baseline, predictions, and theory on top
                    ax_phase.semilogx(f_mes_seg, np.zeros(f_mes_seg.size), 'b--', alpha=0.5)
                    ax_phase.semilogx(f_mes_seg, dP_pred, 'r-')
                    ax_phase.semilogx(f_mes_seg, dP_theo, 'y--')
                    
                    ax_phase.grid(True, which="both", ls="-")
                    ax_phase.set_xlabel('Frequency (Hz)', fontsize=9)
                    
                    # Force strict plain scalar formatting on the shared columns
                    ax_phase.xaxis.set_major_formatter(plt.ScalarFormatter())
                    ax_phase.xaxis.set_minor_formatter(plt.NullFormatter()) 
                    ax_phase.ticklabel_format(style='plain', axis='x')
                    
                    # Calculate boundary limits alongside the exact center frequency
                    f_start_seg = f_mes_seg[0]
                    f_end_seg = f_mes_seg[-1]
                    f_center_seg = (f_start_seg + f_end_seg) / 2
                    
                    # Set the locator to explicitly track left bound, center frequency, and right bound
                    bounds_ticks = [f_start_seg, f_center_seg, f_end_seg]
                    ax_phase.xaxis.set_major_locator(plt.FixedLocator(bounds_ticks))
                    ax_phase.tick_params(axis='x', rotation=0, labelsize=8.5)
                    
                    start_idx = end_idx

                # Lock row headers cleanly to the outer edge of the master figure canvas
                axs[0, 0].set_ylabel(r'$\Delta$ Gain', fontsize=10, weight='bold')
                axs[1, 0].set_ylabel(r'$\Delta$ Phase (deg)', fontsize=10, weight='bold')
                axs[0, 0].legend(loc='upper left', fontsize=8)
                                                                
        self.current_R = R_pred

        if not geo_constraint:
            x_pred = self.solve_x(R_pred)
        
        self.current_x = x_pred
        
        return R_pred, x_pred