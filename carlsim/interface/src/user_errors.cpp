#include <user_errors.h>

#include <carlsim_definitions.h>

#include <stdio.h>
#include <iostream>		// std::cerr, std::cerr, std::endl
#include <stdlib.h>		// exit, EXIT_FAILURE
#include <string>		// std::string

/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///

// simple wrapper to assert a statement and print an error message
void UserErrors::assertTrue(bool statement, errorType errorIfAssertionFails, std::string errorFunc,
								std::string errorMsgPrefix, std::string errorMsgSuffix) {
	if (!statement) {
		throwError(errorFunc,errorIfAssertionFails,errorMsgPrefix,errorMsgSuffix); // display standard error message
	}
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// simple wrapper for displaying standard message per error type
void UserErrors::throwError(std::string errorFunc, errorType error, std::string errorMsgPrefix,
	std::string errorMsgSuffix) {

	std::string errorMsg = errorMsgPrefix;
	
//	std::cerr << "\033[31;1m[USER ERROR " << errorFunc << "] \033[0m" << errorMsgPrefix;
	switch (error) {
	case ALL_NOT_ALLOWED:
		errorMsg += " cannot be ALL."; break;
	case CAN_ONLY_BE_CALLED_IN_MODE:
		errorMsg += " can only be called in mode "; break;
	case CAN_ONLY_BE_CALLED_IN_STATE:
		errorMsg += " can only be called in state "; break;
	case CANNOT_BE_CALLED_IN_MODE:
		errorMsg += " cannot be called in mode "; break;
	case CANNOT_BE_CALLED_IN_STATE:
		errorMsg += " cannot be called in state "; break;
	case CANNOT_BE_CONN_SYN_AND_COMP:
		errorMsg += " cannot be both synaptically and compartmentally connected."; break;
	case CANNOT_BE_CONN_TWICE:
		errorMsg += " connectCompartments is bidirectional: connecting same groups twice is illegal "; break;
	case CANNOT_BE_IDENTICAL:
		errorMsg += " cannot be identical."; break;
	case CANNOT_BE_LARGER:
		errorMsg += " cannot be larger than "; break;
	case CANNOT_BE_NEGATIVE:
		errorMsg += " cannot be negative."; break;
	case CANNOT_BE_NULL:
		errorMsg += " cannot be NULL."; break;
	case CANNOT_BE_POSITIVE:
		errorMsg += " cannot be positive."; break;
	case CANNOT_BE_OFF:
		errorMsg += " cannot not be off at this point."; break;
	case CANNOT_BE_ON:
		errorMsg += " cannot be on at this point."; break;
	case CANNOT_BE_SET_TO:
		errorMsg += " cannot be set to "; break;
	case CANNOT_BE_SMALLER:
		errorMsg += " cannot be smaller than "; break;
	case CANNOT_BE_UNKNOWN:
		errorMsg += " cannot be of type UNKNOWN."; break;
	case CANNOT_BE_ZERO:
		errorMsg += " cannot be zero."; break;
	case FILE_CANNOT_CREATE:
		errorMsg += " could not be created."; break;
	case FILE_CANNOT_OPEN:
		errorMsg += " could not be opened."; break;
	case FILE_CANNOT_READ:
		errorMsg += " could not be read."; break;
	case IS_DEPRECATED:
		errorMsg += " is deprecated."; break;
	case MUST_BE_IDENTICAL:
		errorMsg += " must be identical."; break;
	case MUST_BE_LARGER:
		errorMsg += " must be larger than "; break;
	case MUST_BE_NEGATIVE:
		errorMsg += " must be negative."; break;
	case MUST_BE_POSITIVE:
		errorMsg += " must be positive."; break;
	case MUST_BE_OFF:
		errorMsg += " must be off at this point."; break;
	case MUST_BE_ON:
		errorMsg += " must be on at this point."; break;
	case MUST_BE_IN_RANGE:
		errorMsg += " must be in the range "; break;
	case MUST_BE_SET_TO:
		errorMsg += " must be set to "; break;
	case MUST_BE_SMALLER:
		errorMsg += " must be smaller than "; break;
	case MUST_HAVE_SAME_SIGN:
		errorMsg += " must have the same sign."; break;
	case NETWORK_ALREADY_RUN:
		errorMsg += " cannot be called after network has been run."; break;
	case UNKNOWN_GROUP_ID:
		errorMsg += " is unknown."; break;
	case WRONG_NEURON_TYPE:
		errorMsg += " cannot be called on this neuron type."; break;
	case UNKNOWN:
	default:
		errorMsg += ". An unknown error has occurred."; break;
	}
	errorMsg += errorMsgSuffix;

	CARLSIM_ERROR(errorFunc.c_str(), errorMsg.c_str());
	exit(EXIT_FAILURE); // abort
}