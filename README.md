# DeepFRAG

A method for cancer prediction based on DNA fragmentomics and deep learning



For running the code and reproducing the results, the data must be extracted from data.zip. The contents are unpacked in the directory data with 6 folders (EXP\[0-6]\_\[cancer, healthy]) of entire project data -- fragment size (FS) probability mass function (PMF) profiles, and a folder TrainTestPaths with randomly selected paths to train and test files pointing to EXP... folders. For saving the test results, user has to create directory ./data/Test\_Results with two subdirectories DWT\_DNN and PMF\_DNN, for settings when input information to the deep learning model is discrete wavelet (DWT) coefficients and FS PMF profiles, respectively.



The requirements.txt list all packages in the environment.
