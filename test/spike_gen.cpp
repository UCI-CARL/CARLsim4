#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>
#include <periodic_spikegen.h>
#include <spikegen_from_file.h>
#include <spikegen_from_vector.h>

#include <vector>


TEST(SpikeGen, PeriodicSpikeGenerator) {

}

TEST(SpikeGen, PeriodicSpikeGeneratorDeath) {
	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(0.0);},"");
	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(-10.0);},"");
}

TEST(SpikeGen, SpikeGeneratorFromFile) {

}

TEST(SpikeGen, SpikeGeneratorFromFileDeath) {
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("");},"");
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("thisFile/doesNot/exist.dat");},"");
}

TEST(SpikeGen, SpikeGeneratorFromVector) {

}

TEST(SpikeGen, SpikeGeneratorFromVectorDeath) {
	std::vector<int> emptyVec, negativeVec;
	negativeVec.push_back(0);
	negativeVec.push_back(-1);

	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(emptyVec);},"");
	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(negativeVec);},"");
}