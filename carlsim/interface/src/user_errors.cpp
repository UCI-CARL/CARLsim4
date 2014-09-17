#include <user_errors.h>

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
	
	std::cerr << "\033[31;1m[USER ERROR " << errorFunc << "] \033[0m" << errorMsgPrefix;
	switch (error) {
	case ALL_NOT_ALLOWED:
		std::cerr << " cannot be ALL."; break;
	case CAN_ONLY_BE_CALLED_IN_MODE:
		std::cerr << " can only be called in mode "; break;
	case CAN_ONLY_BE_CALLED_IN_STATE:
		std::cerr << " can only be called in state "; break;
	case CANNOT_BE_CALLED_IN_MODE:
		std::cerr << " cannot be called in mode "; break;
	case CANNOT_BE_CALLED_IN_STATE:
		std::cerr << " cannot be called in state "; break;
	case CANNOT_BE_IDENTICAL:
		std::cerr << " cannot be identical."; break;
	case CANNOT_BE_LARGER:
		std::cerr << " cannot be larger than "; break;
	case CANNOT_BE_NEGATIVE:
		std::cerr << " cannot be negative."; break;
	case CANNOT_BE_NULL:
		std::cerr << " cannot be NULL."; break;
	case CANNOT_BE_POSITIVE:
		std::cerr << " cannot be positive."; break;
	case CANNOT_BE_OFF:
		std::cerr << " cannot not be off at this point."; break;
	case CANNOT_BE_ON:
		std::cerr << " cannot be on at this point."; break;
	case CANNOT_BE_UNKNOWN:
		std::cerr << " cannot be of type UNKNOWN."; break;
	case CANNOT_BE_ZERO:
		std::cerr << " cannot be zero."; break;
	case FILE_CANNOT_CREATE:
		std::cerr << " could not be created."; break;
	case FILE_CANNOT_OPEN:
		std::cerr << " could not be opened."; break;
	case IS_DEPRECATED:
		std::cerr << " is deprecated."; break;
	case MUST_BE_IDENTICAL:
		std::cerr << " must be identical.";
	case MUST_BE_LOGGER_CUSTOM:
		std::cerr << " must be set to CUSTOM."; break;
	case MUST_BE_NEGATIVE:
		std::cerr << " must be negative."; break;
	case MUST_BE_POSITIVE:
		std::cerr << " must be positive."; break;
	case MUST_BE_OFF:
		std::cerr << " must be off at this point."; break;
	case MUST_BE_ON:
		std::cerr << " must be on at this point."; break;
	case MUST_BE_IN_RANGE:
		std:: cerr << " must be in the range"; break;
	case MUST_HAVE_SAME_SIGN:
		std::cerr << " must have the same sign."; break;
	case NETWORK_ALREADY_RUN:
		std::cerr << " cannot be called after network has been run."; break;
	case UNKNOWN_GROUP_ID:
		std::cerr << " is unknown."; break;
	case WRONG_NEURON_TYPE:
		std::cerr << " cannot be called on this neuron type."; break;
	case UNKNOWN:
	default:
		std::cerr << ". An unknown error has occurred."; break;
	}
	std::cerr << errorMsgSuffix << std::endl;
	exit(EXIT_FAILURE); // abort
}