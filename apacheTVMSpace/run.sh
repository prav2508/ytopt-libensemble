#!/bin/bash


python_files=$(ls *.py)

for python_file in $python_files; do
  python $python_file
  echo "Executed $python_file !!"
done