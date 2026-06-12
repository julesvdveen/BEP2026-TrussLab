# TrussLab: an Interactive Introduction to FEM - Code Repository

Welcome to the Git repository for the **TrussLab BEP 2026 project**. For the faculty of mechanical engineering, TrussLab was developed during a Bachelor final project to be used during Continuum Mechanics practise sessions. TrussLab was developed to enable students to build and analyse their own truss structures. The toolbox offers several features: students can construct trusses, deform them by applying custom loads, assign structure constraints (e.g. rigid or single DOF connections), and both analytically estimate and automatically measure the nodal displacements. This last feature enables students to compare algebraic calculations to experimental measurements, and to identify the limitations of the theory they learn during the lectures.

## Content

This repository contains all relevant code developed during the project. Below, a description of the main folders and files is displayed:
- **Research**: this folder contains all research tools and simulation (results). Within, the following can be found:
    - *Jupyter notebooks:* these notebooks, e.g. `experiment_notch_phys_2`, contain either physical (phys) or simulated (no further name additions) experiments. Often, a chronological order can be recognized by observing the version index behind the filenames.
    - *Outdated 'managers'*: old versions of the manager files, still used by several of the experiment notebooks.
    - *beam_libraries*: this folder contains all used beam libraries (datasets containing all beam properties), saved as JSON files. These files can be loaded and modified using the `beam_manager.py`.
    - *myDAQ_logs*: this folder contains all measured data using the NI myDAQ. These logs are loaded within the Jupyter notebooks, to display measurement results. The solver algorithm is run on these measurements, to obtain the final TrussLab measurement output.
    - *LabView*: this folder contains all LabView relevant files. The LabView environment is used to control the NI myDAQ, and is called from within the `myDAQ_manager.py`.

- **TrussLab**: this folder contains the TrussLab module, which can be loaded for practise sessions using the TrussLab hardware. This module contains all required tools for running the full system, including: matching physically built trusses within the digital environment, algebraically calculate nodal displacements, run (electrical) nodal position measurements and more. The module contains so-called 'managers'. A *manager* file regulates all actions and tasks relevant to its name. Below, a short description of each manager is given:
    - `beam_manager.py`: functions within this file are used to initialise new beam libraries (datasets containing all beam properties) and edit them while working on new sessions and experiments.
    - `bridge_manager.py`: this file contains the most important class of the project: the `Bridge` class. During a practise session, a new Bridge object is initialised. A bridge (truss) can be constructed, algebraically analyzed and measurements can be run on the bridge. This same file offers plotting options for obtained data, including tables, (many) graphs and animations.
    - `myDAQ_manager.py`: functions within this file are used to control the NI myDAQ. This manager can be used to load old measurement results, and to call the LabView environment to run new measurements.
    - `student_manager.py`: functions within this file are the only functions directly used by students during practise sessions. The student manager combines the entire TrussLab module into accesible, easy-to-use functions.

- `TrussLab_demo_HTML`: this notebook contains the demonstration which was given to several student groups. These students tried the TrussLab exercises and provided feedback, which can be found in the final BEP article. This file 
- `TrussLabCalibration.ipynb`: this notebook was used to perform final calibration on the individual beams.
- `default.json`: this beam library contains the properties of the final manufactured beam set.

## Reading Recommendations

The TrussLab repository contains over 5000 lines of code, which can be quite challenging to learn to understand. The author's recommend the following reading tips:

- **Try to run the `TrussLab_demo_HTML` first!** This can provide a clear idea of how the TrussLab module is used and which functions are available.
- **Read the BEP article and/or logs!** These files explain the calculation methods in a way easier format than the code does.

If you want to inspect the research files:

- **Especially have a look at files which have `IMPORTANT` in their file name!** These files provide the most important and clear research results. Note that most notebooks contains text blocks which explain the experiment and used simulation methods.
- **Pay attention to outdated experiment files**: these files can be recognized by statements as 'do not run: outdated managers'. All graphed results are already available without running the files.

Good luck! For questions, please send an e-mail to *julesvandervee@tudelft.nl* or *rvanholk@tudelft.nl*.
