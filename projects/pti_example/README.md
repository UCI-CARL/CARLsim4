# Parameter-Tuning Examples for CARLsim

This project demonstrates how CARLsim's parameter-tuning interface (PTI) can be used to optimize the parameters
of an SNN model with the help of an external parameter-tuning tool.

The parameter-tuning methodology used here has two basic components:

 1. The [PTI classes](tools/pti) offer a standardized way to create executable models that receive parameters from
    `std::cin` (or a file) and output a scalar measure of model fitness.

 2. An ECJ parameter file defines and evolutionary algorithm that will use the fitness function defined by the 
    executable model to search for high-performing model configurations.


## How it Works

Running `make` in this directory compiles binaries for each example that take a list of parameters on `std::cin` and 
output scalar fitness values to `std::cout`:

```bash
$ echo "0.1, 0.1, 0.1, 0.1" | ./TuneFiringRates
0.000690725
```

If multiple parameter lists are received, multiple fitness values are returned:
```
$ yes "0.1, 0.1, 0.1, 0.1" | head -n 5 | ./TuneFiringRates 
0.000690403
0.000690441
0.000690906
0.000690644
0.000690801
```