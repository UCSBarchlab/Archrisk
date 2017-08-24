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

uncertainties (v3.0.1)

### Example Usage

###### Covering design space with heterogeous architectural model, quadratic risk function and HPLC application.
'''
python examples/DSE.py --log info --math-model hete --risk-func quad --f 0.999 --c 0.001
'''

###### To create your own model
*Write equations in pure string form in models/math_models.py.
*Enable selection of your model in models/performance_models.py.
*Select and run.

>Happy hacking!
