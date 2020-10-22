
# Supplementary Material for the paper On Testing of Samplers(Paper ID:5110)

  

Barbarik2 is a framework developed to test whether a sampler is epsilon-close or eta-far from a given distribution with confidence greater than 1-delta.

  

The full paper has five Appendices(A-E).

  

Appendices A-D contain the theoretical justification for the theorems and lemmas mentioned in the paper.

Appendix E has the results for the entire set of benchmarks and all the tested samplers.

The results presented in the extended tables of Appendix E can be verified with the code and benchmarks presented in this folder.

  

## Requirements to run the code

  

* Python 2.7

  

To install the required libraries, run:

  

```

pip install -r requirements.txt

```

  

## Getting Started

  

To run with the parameter values used in the paper:

  

```

cp benchmarks/s349_3_2.cnf code/

cd code

python barbarik2.py --eta 1.6 --epsilon 0.1 --delta 0.2 --sampler 2 --seed 42 s349_3_2.cnf

```

  
  

For the command-

  

```

python barbarik2.py --eta ETA --epsilon EPSILON --delta DELTA --sampler SAMPLER-TYPE --seed SEED mycnf.cnf

```

  

ETA takes values in (0,2),

EPSILON takes values in (0,0.33),

DELTA takes values in (0,0.5),

SEED takes integer values, and

  

SAMPLER-TYPE takes the following values:

  

* UniGen2 = 1

* QuickSampler = 2

* STS = 3

  

Note that only UniGen2 shows identical behavior with a fixed seed.

  

### Samplers used

  

In the "samplers" directory, you will find 64-bit x86 Linux compiled binaries for:

  

* UniGen2- an almost-uniform sampler, version 2

* Quick Sampler

* STS
