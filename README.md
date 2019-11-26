## Introduction
Codes for the paper 'Self-organization of action hierarchy and compositionality by reinforcement learning with recurrent networks' (https://arxiv.org/abs/1901.10113) by Dongqi Han, Kenji Doya and Jun Tani.

Cognitive Neurorobotics Research Unit, Okinawa Institute of Science and Technology: https://groups.oist.jp/cnru


## Requirement

Python >= 3.5

### External Libraries

- tensorflow = 1.10.0
- numpy >= 1.16.1
- scipy >= 1.2.1
- gym >= 0.10.9
- matplotlib >= 2.2.2 (only for plotting in Jupyter Notebook)

## To run the codes
Simulation for the consecutive relearning task (wherein phase 1 is exactly the sequential goal reaching task) using ReMASTER can be started by

`python Consecutive_Relearning_Task.py`

If you want to run the alternative models, such as LSTM, you can 

`python Consecutive_Relearning_Task.py --model=LSTM`

And see the following for other options

## optional arguments:
<pre>
--model        The model used, either MTSRNN or LSTM (default: MTSRNN)
--noise        Scale of initial neuronal noise, only works for MTSRNN (default: 0.2)
--singlev      If True, the higher level does not learn the value function with gamma2, only works for MTSRNN (default: False)
--lowstop3     If True, the lower-level synaptic weight will be frozen in phase 3, only works for MTSRNN (default: False)
--seed         Random seed (default: 0)
</pre>

## Data saving
Simulation data of individual episodes (saving every 50 episodes by default) will be saved into a folder `./data`, and the performance curves will be saved into a folder `./perf_data` after simulation finishes. Format of data is ".mat", which can be either read in MATLAB or in python with `scipy.loadmat()`
