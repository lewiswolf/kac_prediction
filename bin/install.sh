#!/bin/bash
# This file downloads datasets from the kac_drumset publication: https://zenodo.org/record/7057219
# It is expected that these datasets will be contained in a directory ./data, which should exist before running this script.

# determine which dataset is to be installed
case $1 in
  2000-convex-polygon-drums-of-varying-size)
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
	if [ -f "$token_file" ]; then
		api_token=$(<$token_file)
	else 
		echo "No access token found..."
		echo "Set one up at https://zenodo.org/account/settings/applications/"
		read -p "Enter your zenodo access token : " api_token
		echo "$api_token" > $token_file
	fi

	# clear directory of all files that are .gitignore and .zenodo
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

	# download dataset from zenodo
	curl --cookie ../.zenodo "https://zenodo.org/record/7057219/files/${dataset}.zip?download=1" --output ${dataset}.zip

	# unzip and copy contents to /data
	echo "organising files..."
	unzip -q ${dataset}.zip
	cp -a ${dataset}/data/. .

	# remove excess files
	rm -rf ${dataset}
	rm ${dataset}.zip
	# and remove any other trash left over
	if [ -d "__MACOSX" ]; then
		rm -rf __MACOSX
	fi
