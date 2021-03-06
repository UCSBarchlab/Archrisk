Archrisk
========

Archrisk is an automated framework to evaluate analytical
architecture models with uncertainties.

It takes string formatted analytical performance equations
and distributional descriptions of uncertain variables, automatically
transforms the equations and injects uncertainties according to
the distributional description and propagates the uncertainties through
to the final responsive metrics.

### Prerequisites

Archrisk depends on the following packages:

numpy (v1.12.1)

scipy ( <= v0.19.1)

mcerp (v0.11)

sympy (v1.0.1)

lmfit (v0.9.9)

Add path to Archrisk to PYTHONPATH.

### Example Usage

###### Exhaustively search design space with heterogeous architectural model, quadratic risk function and HPLC application, using boxcox transformed distributions:
```
python examples/DSE.py --log info --math-model hete --risk-func quad --f 0.999 --c 0.001 --trans
```
>Above command may take up to a few hours to complete.

###### Options that are hard-coded for now:
- In base/eval_functions.py, PERF and FABRIC functions are hard-coded. Change them to other functions defined in models/uncertainty_models.py to run with ground truth distributions.

###### To create and run your own model:
- Write equations in pure string form in models/math_models.py.
- Enable selection of your model in models/performance_models.py.
- Select and run.

>Happy hacking!

### Citation:

Weilong Cui and Timothy Sherwood. 2017. Estimating and understanding architectural risk. In Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO-50 '17). ACM, New York, NY, USA, 651-664. DOI: https://doi.org/10.1145/3123939.3124541

Weilong Cui and Timothy Sherwood. "Architectural Risk" in IEEE Micro: Micro's Top Picks from Computer Architecture Conferences, January-February 2018.
