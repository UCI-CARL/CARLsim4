#! /bin/bash

# choose the parameter value you want to be placed into the csv file
PARAM_VAL=1
echo -e "Using parameter file:" $1
# Read the genome size from the parameter file
GENOME_NUM=$( sed -e '/genome-size/ !d' -e 's/pop.subpop.0.species.genome-size//' -e 's/ \+= \+//' <$1)

# output the genome size for the user as a sanity check
echo -e "Number of genomes in ECJ parameter file is" $GENOME_NUM

# Read the number of individuals per generation
INDI_NUM=$( sed -e '/pop.subpop.0.size/ !d' -e 's/pop.subpop.0.size//' -e 's/ *[tab]*= *[tab]*//' <$1)

# output the number of individuals per generation for the user as a sanity check
echo -e "Number of individuals per generation in ECJ parameter file is" $INDI_NUM

# Generate a single line of test input which is of length $GENOME_NUM
COUNTER=0
ROW_TEXT="$PARAM_VAL,"
while [ $COUNTER -lt $(($GENOME_NUM-1)) ]
do
	COUNTER=$(( $COUNTER + 1 ))
	if [ $COUNTER -lt $(($GENOME_NUM-1)) ]
	then
		ROW_TEXT="$ROW_TEXT $PARAM_VAL,"
	else
		ROW_TEXT="$ROW_TEXT $PARAM_VAL"
	fi
done

COUNTER=0
NEWLINE="\n"
while [ $COUNTER -lt $INDI_NUM ]
do
	if [ $COUNTER -lt $(($INDI_NUM-1)) ]
	then
		CSV=$CSV$ROW_TEXT$NEWLINE
	else
		CSV=$CSV$ROW_TEXT
	fi
	COUNTER=$(( $COUNTER + 1 ))
done

#echo -e $ROW_TEXT "\n"
#echo -e $CSV "\n"
echo -e $CSV > debugInput.csv
#echo -e $COUNTER

# cat debugInput.csv | ./carlsim_tuneFiringRatesECJ
