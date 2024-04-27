#!/bin/bash
# We will download every few days so they aren't too correlated
# over N years so that this turns out to be 1Tb?
dir="./data/"
opts="-N -c -P ${dir}"

year=1987
m=2



if [ "${m}" = 4 ] || [ "${m}" = 6 ] || [ "${m}" = 9 ] ||[ "${m}" = 11 ] ; then
    last_day=30
elif [ "${m}" = 2 ]; then
    if [ "$(((${year}) % 4))" = 0 ]; then
        last_day=29
    else last_day=28
    fi
else last_day=31
fi

month=$(printf "%02d" ${m})
echo "Getting surface files for $year $month"
source ERA5_download_project.sh ${year} ${month} ${last_day}
echo "Subsample in time"
python preprocessing_subsample.py --year ${year} --month ${month} --last_day ${last_day} --remove_files  
echo "Done for ${year} ${month}"


echo "Concatenate all months for ${year}"
python preprocessing_concat_year.py --year ${year} --remove_files
echo DONE

