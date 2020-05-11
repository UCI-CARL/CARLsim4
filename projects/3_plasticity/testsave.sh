#/bin/bash

cd /Users/karl/Documents/Git/CARLsim4/;
make distclean;
make release_nocuda -j4;
make install;
cd projects/3_plasticity;
make distclean;
make nocuda;
./plasticity;