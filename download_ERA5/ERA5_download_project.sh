#!/bin/bash -f
#
# Original c-shell script converted to bash by Laura Mansfield
# c-shell script to download selected files from rda.ucar.edu using Wget
# NOTE: if you want to run under a different shell, make sure you change
#       the 'set' commands according to your shell's syntax
# after you save the file, don't forget to make it executable
#   i.e. - "chmod 755 <name_of_script>"
#
# Experienced Wget Users: add additional command-line flags here
#   Use the -r (--recursive) option with care
dir="./data/"
opts="-N -c -P ${dir}"
#
cert_opt=""
# If you get a certificate verification error (version 1.10 or higher),
# uncomment the following line:
#set cert_opt = "--no-check-certificate"
#
# get year, month, day
year=${1?Error: year?}
month=${2?Error: month?}
last_day=${3?Error: last day of month?}

# download the file(s)

# temperature at 2 m
wget $cert_opt $opts https://data.rda.ucar.edu/ds633.0/e5.oper.an.sfc/${year}${month}/e5.oper.an.sfc.128_167_2t.ll025sc.${year}${month}0100_${year}${month}${last_day}23.nc

# u-component of wind at 10 m
wget $cert_opt $opts https://data.rda.ucar.edu/ds633.0/e5.oper.an.sfc/${year}${month}/e5.oper.an.sfc.128_165_10u.ll025sc.${year}${month}0100_${year}${month}${last_day}23.nc

# v-component of wind at 10 m
wget $cert_opt $opts https://data.rda.ucar.edu/ds633.0/e5.oper.an.sfc/${year}${month}/e5.oper.an.sfc.128_166_10v.ll025sc.${year}${month}0100_${year}${month}${last_day}23.nc


