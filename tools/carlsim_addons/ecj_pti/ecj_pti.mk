# This makefile is a bit of a hack. It makes a bunch of java calls
# I don't want to have to make. It also installs the ecj compnent
# to the correct spot.

ecj_pti_jar := dist/CARLsim-ECJ.jar
all_targets += $(ecj_pti_jar)

# Build the CARLsim-ECJ.jar from nbproject info using ant
ecj_pti: $(ecj_pti_jar)

$(ecj_pti_jar):
	ant jar -Dendorsed.classpath=$(ECJ_DIR)/ecj.jar
