## How to run
From the project root, run the following command in the console:

```python -m project```

## Available commands
* system-ode --- Create neural net for system of ODEs.
* table --- Create neural net for solve table.
* eq --- Create neural network for solve equation.
* quit --- Stop executable.
* save-to-file --- save network to file.
* load-from-file --- load network from file.
* print-info --- print info about network.
* build-plot --- Build plot on interval with answers by network.
* export-solve --- Export solution table with answers by network.
* debug --- Enable/disable debug output.

### Some examples
#### First
```bash
table train_data.csv
build-plot table0 --interval 0 4 --step 0.001

```
#### Second
```bash
save-to-file table0 /networks
load-from-file table0 /networks/table0.txt
```
#### Third
```bash
debug true
eq x**2 --vars x=-4,4,0.01
print-info equation0
build-plot equation0 --interval -5 5 --step 0.005
export-solve equation0 /networks --vars x=-4,4,0.01
```
#### Fourth
```bash
debug true
system-ode 3 --interval 0 50 --points 101
-0.1*y0*y1 y0(0)=2
-0.1*y0*y1 y1(0)=1
0.1*y0*y1 y2(0)=0
build-plot systemode0 --interval 0 50 --step 0.01
export-solve systemode0 ../networks --vars x=0,50,0.5
```