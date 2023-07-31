#!/bin/bash

input="$1"

rm -r slurmLog/err/*
rm -r slurmLog/out/*

# Check if the variable is empty
if [ "$input" == "L" ]; then
    rm logs/large/*
    rm results/large/*
else
    rm logs/extraLarge/*
    rm results/extraLarge/*
fi



