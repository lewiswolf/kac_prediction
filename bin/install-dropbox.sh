#!/bin/bash
# This file downloads datasets from the link provided for the first argument.
# This script is for testing only, and not connnected to any published material.
# It is expected that these datasets will be contained in a directory ./data, which should exist before running this script.


cd data
	# clear directory of all files that are not .gitignore and .zenodo
	for f in *;
	do
		rm -rf $f
	done
	for f in .*;
	do
		case $f in
			.|..|.gitignore|.zenodo)
				continue
				;;
			*)
				rm -rf $f
				;;
		esac
	done

	# get dataset name from link
	dataset=$(basename -s .zip "$1")

	# download the dataset from zenodo
	curl $1 -L --output ${dataset}.zip

	# unzip and copy contents to /data
	echo "Organising files..."
	unzip -q ${dataset}.zip
	cp -a ${dataset}/data/. .

	# remove excess files
	rm -rf ${dataset}
	rm ${dataset}.zip