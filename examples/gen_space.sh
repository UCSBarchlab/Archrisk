mkdir $1
cd $1

time python ../decision_impact.py --log debug --math_model hete --risk_func quad --f 0.999 --c 0.001 $2 &> HPLC.log &
time python ../decision_impact.py --log debug --math_model hete --risk_func quad --f 0.999 --c 0.01 $2 &> HPHC.log &
time python ../decision_impact.py --log debug --math_model hete --risk_func quad --f 0.9 --c 0.001 $2 &> LPLC.log &
time python ../decision_impact.py --log debug --math_model hete --risk_func quad --f 0.9 --c 0.01 $2 &> LPHC.log &
