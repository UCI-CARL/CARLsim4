/*
 * Copyright (c) 2013 Regents of the University of California. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. The names of its contributors may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * *********************************************************************************************** *
 * CARLsim
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 07/13/2013
 */ 

#ifndef __ERROR_CODE_H__
#define __ERROR_CODE_H__

	#define NO_KERNEL_ERRORS		 			0xc00d

	#define NEW_FIRE_UPDATE_OVERFLOW_ERROR1  	0x61
	#define NEW_FIRE_UPDATE_OVERFLOW_ERROR2  	0x62

	#define STORE_FIRING_ERROR_0 				0x54

	#define ERROR_FIRING_0 						0xd0d0
	#define ERROR_FIRING_1 						0xd0d1
	#define ERROR_FIRING_2 						0xd0d2
	#define ERROR_FIRING_3 						0xd0d3

	#define GLOBAL_STATE_ERROR_0  				0xf0f0

	#define GLOBAL_CONDUCTANCE_ERROR_0  		0xfff0
	#define GLOBAL_CONDUCTANCE_ERROR_1			0xfff1

	#define STP_ERROR 							0xf000

	#define UPDATE_WEIGHTS_ERROR1				0x80

	#define CURRENT_UPDATE_ERROR1   			0x51
	#define CURRENT_UPDATE_ERROR2   			0x52
	#define CURRENT_UPDATE_ERROR3   			0x53
	#define CURRENT_UPDATE_ERROR4   			0x54

	#define KERNEL_CURRENT_ERROR0  				0x90
	#define KERNEL_CURRENT_ERROR1  				0x91
	#define KERNEL_CURRENT_ERROR2  				0x92

	#define ICURRENT_UPDATE_ERROR1   			0x51
	#define ICURRENT_UPDATE_ERROR2   			0x52
	#define ICURRENT_UPDATE_ERROR3   			0x53
	#define ICURRENT_UPDATE_ERROR4   			0x54
	#define ICURRENT_UPDATE_ERROR5   			0x55

	#define KERNEL_INIT_ERROR0					0x71
	#define KERNEL_INIT_ERROR1					0x72

	#define POISSON_COUNT_ERROR_0				0x99

#endif
