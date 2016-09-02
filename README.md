# MovieProjections
Classify movie projections and predict future projections

# Arguments
* 1 - datafile, .txt file with your data each column separated by a \t and each row by a new line (\n)
* 2 (optional) - algorithm, default: dt (decision tree) [et, dt, knn, lr, svr]
* 3 (optional) - --debug, enables debug output (print things such as data intermiate states)

# Best results
* Using extra trees regression:
* Best accuracy -> 41%
* Average accuracy -> 33%
* Worst accuracy -> 26%

# Future work
Use discretisation to pre-classify NormalizedAttendance (really low, low, medium, high, really high).
Using this will enable me to apply naïve bayes and other classification algorithms.
