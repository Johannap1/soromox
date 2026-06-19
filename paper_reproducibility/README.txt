This folder aims to be a self-contained storage for the data and plots shown in the SoRoMoX paper. The suggested programming language is Python. The proposed structure is the following:

- example_casestudy1/
	-> code/    -->	 contains the .py files that reads from the data/ folder to process and generate the plots
	-> data/    -->  contains the raw data in .csv, .npy, or .pkl format that are processed to generate the plots
	-> images/  -->  contains the plots of the current case study in .pdf format preferably

- example_casestudy2/
	-> code/
	-> data/
	-> images/

- ...

- final_figures/    -->  contains the final plots used in the paper
		