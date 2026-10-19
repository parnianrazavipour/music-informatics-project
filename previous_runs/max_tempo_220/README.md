# Earlier run: tempo range 60–220 BPM

This folder keeps the run with the updated rhythm extraction and a tempo range of 60–220 BPM (the values set in `src/Rythm_extraction.py`). It has the four notebooks that use the rhythm features, with their outputs, and all their result files, in the same folder structure as the main project.

The main run now uses a tempo range of 60–280 BPM. With a limit of 220 BPM, fast dances stretched by +10% (for example Quickstep at about 204–211 BPM, which becomes 224–232 BPM) go over the limit, and the tempo estimate is forced to jump to another tempo. With 280 BPM the tempo estimate follows the stretch for 89–95% of the clips at every stretch rate, against 75–91% with 220 BPM.

The notebooks here are a record of that run. Their paths point to the main project folder, so they are not meant to be run from this folder.
