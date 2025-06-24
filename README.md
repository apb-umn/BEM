This repo will contain the files for

Approximating Transition Dynamics with Discrete Choice


SS
The ss is now solved with EGM
using files 
OCModelEGMInputs.jl that defines the struct
OCModelEGM.jl that has all the solver functions
OCModelEGM_driver.jl thats the driver file

Transition
OCModelEGM_transition.jl and OCModelEGM_transition_driver.jl do transitions

Optimal Tau
OCModelEGM_opttaxmpi.jl and  OCModelEGM_opttaxmpi_driver.jl do the grid search for optimal taub

run_all_results.jl runs all the cases and stores data
make_data_for_draft.jl creates inputs for Tables and Figures in the main body
summarize_opt.jl 

