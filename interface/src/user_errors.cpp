#include <user_errors.h>

#include <iostream>		// cout, endl
#include <stdlib.h>		// exit, EXIT_FAILURE


/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///

// simple wrapper to assert a statement and print an error message
void UserErrors::userAssert(bool statement, errorType errorIfAssertionFails, std::string errorFunc) {
	if (!statement) {
		throwError(errorFunc,errorIfAssertionFails); // display standard error message
	}
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// simple wrapper for displaying standard message per error type
void UserErrors::throwError(std::string errorFunc, errorType error) {
	std::cout << "USER ERROR: " << errorFunc;
	switch (error) {
		case NETWORK_ALREADY_RUN:
			std::cout << " cannot be called after network has been run.";
			break;
		case UNKNOWN_GROUP_ID:
			std::cout << " is unknown.";
			break;
		case WRONG_NEURON_TYPE:
			std::cout << " cannot be called on this neuron type.";
			break;
		case UNKNOWN:
		default:
			std::cout << ". An unknown error has occurred.";
			break;
	}
	std::cout << std::endl;
	exit(EXIT_FAILURE); // abort
}