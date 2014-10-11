README for CARLsim-ECJ Project
-------------------------------------------------------------------------------

Here are some notes for the CARLsim-ECJ Project.

### SOURCE CODE DIRECTORY DESCRIPTION

<pre>
  Main
directory
    ├── AUTHORS
    ├── C++.gitignore
    ├── README.md
    ├── src
    │   └── ecjapp
    │       ├── CARLsimEC.java
    │       ├── eval
    │       │   ├── problem
    │       │   │   ├── CARLsimController.java
    │       │   │   ├── CARLsimProblem.java
    │       │   │   ├── PopulationToFile.java
    │       │   │   └── RemoteLoginInfo.java
    │       │   ├── SimpleGroupedEvaluator.java
    │       │   └── SimpleGroupedProblemForm.java
    │       └── util
    │           ├── Misc.java
    │           └── Option.java
    └── test
    	├── CARLsim-app
    	│   └── Makefile
    	└── ecjapp
            ├── doubles
            │   ├── TestIndividual.java
            │   ├── TestSimpleGroupedProblem.java
            │   └── TestSimpleProblem.java
            ├── eval
            │   └── SimpleGroupedEvaluatorTest.java
            ├── PopulationToFileTest.java
            └── util
            	└── MiscTest.java
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
