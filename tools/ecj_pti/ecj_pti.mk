# This makefile is a bit of a hack. It makes a bunch of java calls
# I don't want to have to make. It also installs the ecj compnent
# to the correct spot.

ecj_pti_jar := dist/CARLsim-ECJ.jar
all_targets += $(ecj_pti_jar)
install_targets += install_ecj_pti
output_files += build dist *.log *.stat results/*.dat

PHONY: ecj_pti install_ecj_pti
# Build the CARLsim-ECJ.jar from nbproject info using ant
ecj_pti: $(ecj_pti_jar)

$(ecj_pti_jar):
	ant jar -Dendorsed.classpath=$(ECJ_DIR)

install_ecj_pti: $(ecj_pti_jar)
	@test -d $(ECJ_PTI_DIR) || mkdir -p $(ECJ_PTI_DIR)
	@test -d $(ECJ_PTI_DIR)/lib || mkdir -p $(ECJ_PTI_DIR)/lib
	@install -m 0755 $(ecj_pti_jar) $(ECJ_PTI_DIR)/lib

