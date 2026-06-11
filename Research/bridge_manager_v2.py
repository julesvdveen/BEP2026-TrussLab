import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution, minimize

class Bridge:
    """
    Bridge class for bridge structure analysis.

    Attributes:

        library (dict): beam library applied for bridge construction.
        topology (list): topology list describing bridge structure.
        n (int): number of nodes/joints for given topology.
        k (int): number of (unique) beams/vertices in the structure.

    Methods:
        NAM(w, R_ref, R_vec=None): returns (complex) Nodal Admittance Matrix (NAM) for given
                                 frequency, reference resistance and beam resistance vector.
        solve_U(w, U_in, R_ref, R_vec=None): returns voltage vector for all nodes n for given
                                             frequency, reference resistance and beam
                                             resistance vector.
    """

    def __init__(self):

        self.library = None # beam library used for bridge construction.
        self.topology = None # Topology list
        self.n = None # Number of nodes/joints.
        self.k = None # Number of unique beams (in library).
    
    def construct(self, library : dict, topology : list):

        self.__init__()

        k_lib = len(library)
        k_top = len(topology)
        if k_lib != k_top:
            raise Exception(f'topology list dimension {k_top} does not match library dimension {k_lib}')
        self.k = k_top

        self.library = library
        self.topology = topology

        max_list = np.zeros(len(topology))
        for i in range(len(topology)):
            max_list[i] = max(topology[i])
        self.n = int(np.max(max_list)) + 1
    
    def solve_U(self, f_arr, R_ref, U_in=1, R_vec=None, R_int=None, noise : bool=False, **kwargs):

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

        if R_int is None:
            NAM[:,-1,-1] += 1/R_ref
        else:
            NAM[:,-1,-1] += 1/R_ref + 1/R_int # parallel

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
    
    def initial_response(self, f_arr, R_ref, R_int=None, plot=False):

        U_out = self.solve_U(f_arr, R_ref, R_int=R_int)

        if plot:

            U_out_mag = np.abs(U_out)
            U_out_deg = np.degrees(np.angle(U_out))

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
        
        return U_out
    
    def predict_R(self, R_ref, R_int=None, myDAQ=False, f_logrange=(2, np.log10(2e4)), freqsteps=1500, resonance_sweep=False, 
                  max_iter=500, error_plot=False, response_comparison_plot=False, **kwargs):

        f_start, f_stop = f_logrange
        f_arr = np.logspace(f_start, f_stop, freqsteps)
        
        f_base_targets = None 
        f_diff_targets = None

        abs_noise = None
        sigma = None
        rng = None
        
        if not myDAQ:
            R_def = kwargs.get('R_def', None) 

        noise = kwargs.get('noise', False)

        if not myDAQ and noise:
            abs_noise = kwargs.get('abs_noise', 0.005) # 5 mV
            sigma = kwargs.get('sigma', 0.005) # 0.5% of signal strength
            rng = kwargs.get('rng')
            if rng is None: rng = np.random.default_rng()

        if resonance_sweep:

            f_arr_base = np.logspace(f_start, f_stop, 15000)
            
            U_out_base = self.initial_response(f_arr_base, R_ref, R_int=R_int)
            mag_base = np.abs(U_out_base)

            # Find base peaks
            base_peaks, _ = find_peaks(mag_base, prominence=0.01)
            base_dips, _ = find_peaks(-mag_base, prominence=0.01)
            base_indices = np.unique(np.concatenate((base_peaks, base_dips)))

            # --- SENSITIVITY DIFFERENCE CALCULATION (Noise-Free) ---
            R_nominal = [beam.R_0 for beam in self.library.values()]
            R_perturbed = [R * 1.10 for R in R_nominal]
            
            U_out_sens = self.solve_U(f_arr_base, R_ref, R_vec=R_perturbed, noise=False)
            mag_sens = np.abs(U_out_sens)
            
            dU_sens = np.abs(mag_base - mag_sens)

            # Find sensitivity features
            sens_prom = kwargs.get('sens_prominence', 1e-4) 
            diff_peaks, _ = find_peaks(dU_sens, prominence=sens_prom)

            # Store separate arrays for the plotting block
            f_base_targets = f_arr_base[base_indices]
            f_diff_targets = f_arr_base[diff_peaks]
            
            print(f"Targeting {len(f_base_targets)} base features at (Hz): {np.round(f_base_targets, 1)}")
            print(f"Targeting {len(f_diff_targets)} diff features at (Hz): {np.round(f_diff_targets, 1)}")

            # Combine for the solver
            all_indices = np.unique(np.concatenate((base_indices, diff_peaks)))
            f_sweep = f_arr_base[all_indices]

            num_peaks = len(f_sweep)
            if num_peaks > 0:
                points_per_peak = freqsteps // num_peaks 
                targeted_f_list = []
                
                # --- DYNAMIC LOG-SCALING df LOGIC ---
                static_df = kwargs.get('df', None) 
                rel_df = kwargs.get('rel_df', 0.05) # Default to +/- 5% of the center frequency

                for f in f_sweep:
                    # If df is None, scale the bandwidth proportionally to the frequency
                    relative_df = static_df if static_df is not None else (f * rel_df)
                    
                    f_lower = max(f - relative_df, 1e-3) 
                    f_upper = f + relative_df
                    cluster = np.linspace(f_lower, f_upper, points_per_peak)
                    targeted_f_list.append(cluster)
                    
                f_arr = np.unique(np.concatenate(targeted_f_list))
            
            if myDAQ:
                raise Exception('Resonance sweep mode is not compatible with myDAQ measurements yet.')
            else:
                U_out_mes = self.solve_U(f_arr, R_ref, R_vec=R_def, R_int=R_int, noise=noise, abs_noise=abs_noise, sigma=sigma, rng=rng)

        else:
            
            if myDAQ:
                f_arr = kwargs.get('f_arr_mes', None)
                U_out_mes = kwargs.get('U_out_mes', None)
                if U_out_mes is None:
                    raise Exception('When using myDAQ, the measured output response must be provided as arguments f_arr_mes and U_out_mes.')
            else:
                f_arr = np.logspace(f_start, f_stop, freqsteps)
                U_out_mes = self.solve_U(f_arr, R_ref, R_vec=R_def, R_int=R_int, noise=noise,
                                          abs_noise=abs_noise, sigma=sigma, rng=rng)
            
        def cost(R):
            U_out_pred = self.solve_U(f_arr, R_ref, R_vec=R, R_int=R_int)
            w = 1  # Could implement frequency weighting here if desired
            U_mag_err = np.abs(U_out_mes - U_out_pred)
            J = np.sqrt(np.mean((w * U_mag_err)**2)) 
            return J

        bounds = [(1.0, 1e4) for i in range(self.k)]

        res_global = differential_evolution(cost, bounds, maxiter=max_iter, tol=1e-8, seed=42)
        if not res_global.success:
            raise Exception(f'The global optimization (differential evolution) did not converge: {res_global.message}')
        
        res = minimize(cost, res_global.x, method='SLSQP', bounds=bounds, options={'ftol': 1e-8})
        if not res.success:
            raise Exception(f'The local optimization (SLSQP) did not converge: {res.message}')
        R_pred = res.x

        if error_plot:
            
            R_var = kwargs.get('R_var', [0])
            R_def = kwargs.get('R_def', None)

            Rdev = 300.0
            steps = 75

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

                costs = [cost(R_temp[:,i]) for i in range(steps)]

                plt.figure(figsize=(8, 5))
                plt.plot(r_space, costs, label=f'Cost $J$')
                plt.axvline(R_pred[idx], color='purple', linestyle='--', label=f'Predicted $R_{{{list(self.library.keys())[idx]}}}$')
                if R_def != None:
                    plt.axvline(R_def[idx], color='g', linestyle='--', label=f'Correct $R_{{{list(self.library.keys())[idx]}}}$')
                plt.xlabel(f'Resistance $R_{{{list(self.library.keys())[idx]}}}$')
                plt.ylabel('Cost $J$ [V]')
                plt.title('RMSE Cost')
                plt.legend()
                plt.grid(True)

            if len(R_var) == 2:
                idx = R_var[0]
                idy = R_var[1]

                if R_def == None:
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

                R_temp = np.full((X_flat.size, self.k), R_pred)
                R_temp[:, idx] = X_flat
                R_temp[:, idy] = Y_flat

                costs_flat = np.array([cost(R_temp[i, :]) for i in range(len(X_flat))])
                costs = costs_flat.reshape(X.shape)

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
                
                fig.colorbar(cntr, ax=ax, label='Cost $J$ (V)')

        if response_comparison_plot:

            U_out_base = self.solve_U(f_arr, R_ref, R_vec=None)
            U_out_base_mag = np.abs(U_out_base) # Gain
            U_out_base_deg = np.degrees(np.angle(U_out_base))

            U_out_pred = self.solve_U(f_arr, R_ref, R_vec=R_pred)
            U_out_pred_mag = np.abs(U_out_pred) # Gain
            U_out_pred_deg = np.degrees(np.angle(U_out_pred))

            U_out_theory = self.solve_U(f_arr, R_ref, R_vec=R_def)
            U_out_theory_mag = np.abs(U_out_theory) # Gain
            U_out_theory_deg = np.degrees(np.angle(U_out_theory))

            dU_pred_mag = U_out_base_mag - U_out_pred_mag # Gain Difference
            dU_pred_deg = U_out_base_deg - U_out_pred_deg

            dU_theory_mag = U_out_base_mag - U_out_theory_mag # Gain Difference
            dU_theory_deg = U_out_base_deg - U_out_theory_deg

            fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

            axs[0].semilogx(f_arr, np.full(f_arr.size, 0), color='b', linestyle='--', alpha=0.5, label='Base')
            axs[0].semilogx(f_arr, dU_pred_mag, color='r', label='Predicted')
            axs[0].semilogx(f_arr, dU_theory_mag, color='y', linestyle='--', label='Theory')

            axs[1].semilogx(f_arr, np.full(f_arr.size, 0), color='b', linestyle='--', alpha=0.5, label='Base')
            axs[1].semilogx(f_arr, dU_pred_deg, color='r', label='Predicted')
            axs[1].semilogx(f_arr, dU_theory_deg, color='y', linestyle='--', label='Theory')

            if kwargs.get('include_mes', False):

                U_out_mes_mag = np.abs(U_out_mes) # Gain
                U_out_mes_deg = np.degrees(np.angle(U_out_mes))

                dU_mes_mag = U_out_base_mag - U_out_mes_mag # Gain Difference
                dU_mes_deg = U_out_base_deg - U_out_mes_deg
                
                axs[0].semilogx(f_arr, dU_mes_mag, color='g', label='Measured')
                axs[1].semilogx(f_arr, dU_mes_deg, color='g', label='Measured')
            
            axs[0].set_ylabel(r'$\Delta$ Gain', fontsize = 15)
            axs[0].legend()
            axs[0].grid(True, which="both", ls="-")
                

            axs[1].set_xlabel('Frequency (Hz)', fontsize = 15)
            axs[1].set_ylabel(r'$\Delta$ Phase (deg)', fontsize = 15)
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

        return R_pred