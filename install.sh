#!/bin/bash

if [ ! -d miniconda ]; then
	make
	make install
fi

source miniconda/bin/activate plan