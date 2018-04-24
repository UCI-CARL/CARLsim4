README for CARLsim-ECJ Project
-------------------------------------------------------------------------------

Here are some notes for the CARLsim-ECJ Project.

### Quickstart

The current version of the CARLsim paramter-tuning framework uses Evolutionary
Computations in Java (ECJ) (version 22 or greater is required). For information
on how to install ECJ, please go [here](http://cs.gmu.edu/~eclab/projects/ecj/).



1) Set the ECJ_JAR and ECJ_PTI_DIR variables in ~/.bashrc.
   NOTE: the ECJ_JAR includes the name of the ECJ jar file.
	 EXAMPLE: if the ECJ jar file is found in /opt/ecj/jar, and named
	 ecj.22.jar, then:
	 ECJ_JAR=/opt/ecj/jar/ecj.22.jar.

2) Change the current directory to ’tools/ecj_pti’.

3) Type ‘make clean && make && make install’

4) Refer to http://uci-carl.github.io/CARLsim4/ch10_ecj.html and http://uci-carl.github.io/CARLsim4/tut7_pti.html for installation and
   how to use CARLsim and ECJ to tune SNNs.

TO UNINSTALL:
CARLsim: Remove the folder where you installed the CARLsim ECJ PTI library. This folder is located in $(ECJ_PTI_DIR).

Type ‘make help’ for additional information.


### SOURCE CODE DIRECTORY DESCRIPTION

<pre>
├── AUTHORS
├── build.xml
├── C++.gitignore
├── ecj_pti.mk
├── Makefile
├── nbproject
│   ├── build-impl.xml
│   ├── genfiles.properties
│   ├── project.properties
│   └── project.xml
├── README.md
├── results
├── src
│   ├── CARLsim-app
│   ├── doc-files
│   ├── ecjapp
│   ├── izk
│   └── overview.html
├── src.mk
├── test
│   ├── CARLsim-app
│   └── ecjapp
└── user.mk
</pre>


* Main directory - contains the Makefile, documentation files, and other
directories.

* src - contains source code for both the ECJ component (found in ecjapp)
and the CARLsim component.

* test - contains testing code for both the ECJ and CARLsim components.
The CARLsim testing framework code is found in test/CARLsim-app while
the ECJ testing framework code is found in test/ecjapp.

### Using the CARLsim testing framework

CARLsim uses the googletest framework v1.7 for testing. For more information
on googltest please visit the website at https://code.google.com/p/googletest/.
For a quick primer, visit: https://code.google.com/p/googletest/wiki/Primer.

To use the googletest framework, you must first download googletests and
point to the googletest directory. One must link to the correct library (either
gtest.a or gtest_main.a) and include the correct headers during compilation.
Please see the test/CARLsim-app/Makefile for examples of how to compile your
tests.
