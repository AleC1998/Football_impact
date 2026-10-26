# Football and COVID-19 code

- `figures/`: Figure 2, Figure 3, the rolling stadium-effect plots, the year comparison, and the two catchment maps.
- `fit.py`: the fitted home/TV model and bootstrap; it calls `permutation_tests.py` for the paper's permutation tests.
- `model_core.py`: input readers and baseline mathematics required by `fit.py`, not an alternative fit.
- `S.py`: fan-share distribution.
- `W.py`: the stadium-catchment specification used in the paper.
- `data/`: only the supplied public S/W inputs and allocations.

Install `requirements.txt` with Python 3.11 or later. Run scripts from this folder. Results go into `results/`.

For the fit and figures, replace the `example_*.csv` filenames with your own appropriate files. Each loading section states the expected columns and units. No observations, fitted estimates or example data records are included. The fit's outcome/fixture loading paths are in `model_core.py`; the TV-status loading path is in `fit.py`.

Run `python fit.py` for one outcome, `python S.py`, `python W.py`, or the appropriate `python figures/<script>.py`. `python fit.py --build-only` checks the design; `--resume` reuses compatible fit checkpoints; `--bootstrap-only` omits permutation tests. The paper defaults are 9,999 bootstrap draws, 999 permutations, and an assumed audience of 15 million.

The supplied catchment maps use the original 317-district allocations. The fitted study and S/W calculations use April 2020 geography with 314 English districts. The plotting calculations and the retained fitting/S/W formulas are unchanged; unused fitting variants, report pipelines and batch wrappers are omitted.
