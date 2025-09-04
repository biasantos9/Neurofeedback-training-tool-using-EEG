I. Data Set

The dataset we chose is highly approximately to the procedure we adopt on EEG acquisition using the Muse Headband device.
It is related to four participants and the signal is acquired in three different states: neutral, relaxed and concentrated.
We chose to initially use this dataset to process a signal with less artifacts and errors, to verify our strategy so that we perform the same steps on our own EEG signal acquired.
Similarly to our project, they used a Muse Headband with four channels (TP9, AF7, AF8 and TP10) to do the acquisition, making the results very approximate to our goal.
The acquisition was made during 60 seconds in each file [1]. The following dataset was used: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state.

The dataset is composed of different files named subject{x}-{state}-1.csv, each one related to one participant (from 1 to 4) and one an acquisition related to one state (neutral, relaxed or concentrated).
The acquisition protocol used in the signal acquisition of the dataset was similar to the one adopted in our group's EEG acquisition.
For a neutral mental state, a test was carried out without any stimuli - being carried out first, before other test or stimulus, to avoid lasting effects of a relaxed or concentrated mental state.
In the relaxation task, the condition implemented for the study subjects was to listen to low-speed music and sound effects designed to aid meditation, while receiving instructions for muscle relaxation and rest.
Finally, for concentration, the subjects were instructed to perform the "cup game" in which a ball was hidden under one of three cups, which were then changed positions,
with the aim of following the path of the ball and matching its final location.

II. Project idea

The main objective of the project is to build a neurofeedback system that analyzes multiple EEG features during a relaxation interval and detects a state of concentration/fatigue,
distracting the user with a vibration/light stimulus.

III. Software

To accomplish this, the project used Python language and an auxiliar Muse Headband device for signal acquisition.

V. References

[1] https://github.com/jordan-bird/eeg-feature-generation
