This file explains how to run the our algorithm.

To run the algorithm the folder datasets and the folder functions are required. 
Then the main function can be executed which has to be in the same folder. 

The parameters for the run and the dataset to be used can be specified at the bottom of the main fanction under the section parameters. 
The most important ones are the maximum number of iterations which can be set at iterations.
In order to specify the file the file name has to be given to Solver as a parameter, possible arguments are file_path1 to file_path9.
After choosing an input file the corresponding reference point has to be chosen in the intitlization part and passed to the solver instance.
Possible arguments are reference_points[0] to reference_points[8].
The code will still run with the wrong reference point but will generate nonsensical outputs for the hypervolume.

The other parameters affect the inner workings of the algorithm and have been set to reasonable values.
For further information on these please refer to the project report.
Have fun :) 