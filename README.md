# FFR SSNMF Feasibility Project

This project demonstrates how a source separation non-negative matrix factorization (SSNMF) algorithm can be used to separate the frequency-following response (FFR) from noise.

- `main.ipynb` is the main script of this project.
- `spectrogram_data.npy` is a sample recording that contains a series of 11 amplitude spectrograms, when 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, and 8000 sweeps are randomly selected from a pool of 8000 EEG sweeps, in reponse to an English vowel /i/ with a rising frequency contour. This sample recording is obtained from a normal-hearing participant. Each spectrogram is a 1000 x 201 (frequency x time) matrix in nanovolts. Thus, `spectrogram_data.npy` is a 1000 x 201 x 11 (number of frequency components x number of time bins x number of spectrograms) array in numpy.

<<<<<<< HEAD
Reference: Jeng, F.-C., Lin, T.-H., Hart, B., N., Montgomery-Regan, K., & McDonald, K. (2023) Non-negative matrix factorization improves the efficiency of recording frequency-following responses in normal-hearing adults and neonates. Int J Audiol, 62(7), 688-698. http://www.doi.org/10.1080/14992027.2022.2071345 (open access)
=======
Reference: **Jeng, F.-C.**, Lin, T.-H., Hart, B., N., Montgomery-Regan, K., & McDonald, K. (2022, published online ahead of print) Non-negative matrix factorization improves the efficiency of recording frequency-following responses in normal-hearing adults and neonates. _Int J Audiol_, http://www.doi.org/10.1080/14992027.2022.2071345 (open access)
>>>>>>> 0aa4554052850828f9a481760a4b885f24e5c49b

This package is developed by Fuh-Cherng Jeng at Ohio University and Tzu-Hao Lin at Academia Sinica. This package is distributed to facilitate learning, but without any warranty. The user is free to use, modify, and re-distribute this package under the terms of an MIT license.
