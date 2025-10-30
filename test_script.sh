#!/bin/bash
#SBATCH --output=output.%j.test.out
#SBATCH --job-name=FIRST_SCRIPT_TEST_RUN

python test_runfile.py