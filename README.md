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

scipy (v0.18.1)

mcerp (v0.11)

sympy (v1.0.1)

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

###### Citation:

@inproceedings{Cui:2017:EUA:3123939.3124541,
 author = {Cui, Weilong and Sherwood, Timothy},
 title = {Estimating and Understanding Architectural Risk},
 booktitle = {Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture},
 series = {MICRO-50 '17},
 year = {2017},
 isbn = {978-1-4503-4952-9},
 location = {Cambridge, Massachusetts},
 pages = {651--664},
 numpages = {14},
 url = {http://doi.acm.org/10.1145/3123939.3124541},
 doi = {10.1145/3123939.3124541},
 acmid = {3124541},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {architecture modelling, core selection, random variable, uncertainty},
}
