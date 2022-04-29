- In order to run the q2main program
	1. change the root variable in q2main.py to the file path of the folder where CSV files are locates.
	2. make sure that you installed Python3, numpy, pandas and matplotlib.
	3. run the program with the command "python3 q2main.py"
- If the CSV file names are different, change them
- The program prints out the accuracy and time measurement of  the non removed feature version of the model. Then, it starts 
removing each future one by one while printing out each of them's time complexity and accuracy. After finishing the first step of
the elimination, it pops up a plot showing that how the removal of one future resulted in which accuracy score. This continues
in each step until there is no change in the max accuracy. At last, program prints which features will be removed in order to
achieve higher accuracy and prints the accuracy after removing those features.  

-In order to run the q31main program
	1. change the root variable in q31main.py to the file path of the folder where CSV files and other necessary files are located.
	2. make sure that you installed Python3, numpy and pandas.
	3. run the program with the command "python3 q31main.py"
- If the CSV file names are different, change them
-The program prints out the accuracy result with the confusion matrix of the model

-In order to run the q33amain program
	1. change the root variable in q33amain.py to the file path of the folder where CSV files and other necessary files are located.
	2. make sure that you installed Python3, numpy and pandas.
	3. run the program with the command "python3 q33amain.py"
- If the CSV file names are different, change them.
-The program prints out the training time for each step and its accuracy result with the confusion matrix of the model.