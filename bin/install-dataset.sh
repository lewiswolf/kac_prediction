#!/bin/bash
# This file downloads datasets from the kac_drumset publication: https://zenodo.org/record/7057219
# It is expected that these datasets will be contained in a directory ./data, which should exist before running this script.

# determine which dataset should be installed
case $1 in
  2000-convex-polygonal-drums-of-varying-size)
    dataset=$1
    ;;
  5000-circular-drums-of-varying-size)
    dataset=$1
    ;;
  5000-rectangular-drums-of-varying-dimension)
    dataset=$1
    ;;
  *)
	echo "Argument 1 is not a valid dataset."
	exit 0
    ;;
esac

cd data
	# configure zenodo token
	token_file=".zenodo"
	if [ ! -f "$token_file" ]; then
		echo "No access token found..."
		echo "Set one up at https://zenodo.org/account/settings/applications/"
		read -p "Enter your zenodo access token: " api_token
		echo "$api_token" > $token_file
	fi

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

	# download the dataset from zenodo
	curl --cookie ${token_file} "https://zenodo.org/records/7274474/files/${dataset}.zip?download=1" --output ${dataset}.zip

	# unzip and copy contents to /data
	echo "Organising files..."
	unzip -q ${dataset}.zip
	cp -a ${dataset}/data/. .

	# remove excess files
	rm -rf ${dataset}
	rm ${dataset}.zip