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
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARL/CARLsim/
 * Ver 10/09/2013
 */ 

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#if (WIN32 || WIN64)
	#include <windows.h>
	#include <winsock2.h>
	#include <ws2tcpip.h>
	#include <iphlpapi.h>

	#define SOCKET_LAST_ERROR WSAGetLastError()
	#define SOCKET_CLEAN_UP WSACleanup()
	#define CLOSE_SOCKET(a) closesocket(a)
	#define SHUT_SEND SD_SEND
#else
	#include <unistd.h>
	#include <errno.h>
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <netdb.h>
	#include <arpa/inet.h>
	#include <pthread.h>

	#define SOCKET int
	#define SOCKADDR_IN struct sockaddr_in
	#define SOCKADDR struct sockaddr
	#define ADDRINFO struct addrinfo
	#define INVALID_SOCKET (-1)
	#define SOCKET_ERROR (-1)
	#define SOCKET_LAST_ERROR errno
	#define CLOSE_SOCKET(a) close(a)
	#define SOCKET_CLEAN_UP
	#define SHUT_SEND SHUT_WR

#endif
#include <stdio.h>

#include "snn.h"
#include "server_client.h"

#define N 1000

#define DEFAULT_BUFLEN 128
#define DEFAULT_TCPIP_PORT "27016"
#define DEFAULT_UDP_PORT 27000

#define BUF_LEN 128

#if (WIN32 || WIN64)
	#pragma comment(lib, "Ws2_32.lib")
#endif

typedef struct carlsim_service_config_t {
	volatile bool execute;
	volatile bool run;
	volatile bool display;
	SOCKADDR_IN clientAddr;
} CARLsimServiceConfig;


class SpikeController: public SpikeMonitor, SpikeGenerator {
private:
	SOCKET dataSocket;
	SOCKADDR_IN clientAddr;
	unsigned int buf[BUF_LEN];
	int bufPos;

public:
	SpikeController(SOCKET ds, SOCKADDR_IN cd) {
		dataSocket = ds;
		clientAddr = cd;
		bufPos = 0;
	}

	// nextSpikeTime is called every one second (simulation time)
	unsigned int nextSpikeTime(CpuSNN* s, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		return 0xFFFFFFFF;
	}

	// update is called every one second (simulation time)
	void update(CpuSNN* s, int grpId, unsigned int* Nids, unsigned int* timeCnts)
	{
		int pos = 0;
		int numByteSent;
		unsigned int gId = grpId << 24; // support 256 groups

		for (int t = 0; t < 1000; t++) {
			for(int i = 0; i < timeCnts[t]; i++, pos++) {
				unsigned int time = t + s->getSimTime() - 1000;
				unsigned int id = Nids[pos];

				buf[bufPos] = time;
				buf[bufPos + 1] = gId | id;
				bufPos += 2;

				// send out data if buffer is full
				if (bufPos >= BUF_LEN) {
					int numByteSent = sendto(dataSocket, (char*)buf, BUF_LEN * sizeof(unsigned int), NULL, (SOCKADDR*)&clientAddr, sizeof(SOCKADDR_IN));
					//printf("send out %d bytes udp data\n", numByteSent);

					bufPos = 0;
				}
			}
		}

		// send out the rest of data
		if (bufPos > 0) {
			numByteSent = sendto(dataSocket, (char*)buf, bufPos * sizeof(unsigned int), NULL, (SOCKADDR*)&clientAddr, sizeof(SOCKADDR_IN));
			//printf("send out %d bytes udp data\n", numByteSent);
		}

		bufPos = 0;
	}
};

#if (WIN32 || WIN64)
DWORD WINAPI service(LPVOID lpParam)
#else
void *service(void *lpParam)
#endif
{
	CpuSNN* s;
	CARLsimServiceConfig* csc = (CARLsimServiceConfig*)lpParam;
	
	SOCKET dataSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	SpikeController* spikeCtrl = new SpikeController(dataSocket, csc->clientAddr);

	// create a network
	s = new CpuSNN("global", CPU_MODE);

	int g1 = s->createGroup("excit", N * 0.8, EXCITATORY_NEURON);
	s->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int g2 = s->createGroup("inhib", N * 0.2, INHIBITORY_NEURON);
	s->setNeuronParameters(g2, 0.1f,  0.2f, -65.0f, 2.0f);

	int gin = s->createSpikeGeneratorGroup("input", N * 0.8, EXCITATORY_NEURON);

	// make random connections with 10% probability
	s->connect(g2, g1, "random", -2.0f/100, -2.0f/100, 0.1f, 1, 1, SYN_FIXED);
	// make random connections with 10% probability, and random delays between 1 and 20
	s->connect(g1, g2, "random", +2.5f/100, 5.0f/100, 0.1f,  1, 20, SYN_PLASTIC);
	s->connect(g1, g1, "random", +4.0f/100, 10.0f/100, 0.1f,  1, 20, SYN_PLASTIC);

	// one-to-one connection
	s->connect(gin, g1, "one-to-one", +20.0f/100, 20.0f/100, 1.0f,  1, 20, SYN_FIXED);

	float COND_tAMPA = 5.0, COND_tNMDA = 150.0, COND_tGABAa = 6.0, COND_tGABAb = 150.0;
	s->setConductances(ALL, true, COND_tAMPA, COND_tNMDA, COND_tGABAa, COND_tGABAb);

	// here we define and set the properties of the STDP. 
	float ALPHA_LTP = 0.10f/100, TAU_LTP = 20.0f, ALPHA_LTD = 0.08f/100, TAU_LTD = 40.0f;	
	s->setSTDP(g1, true, false, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
	s->setSTDP(g2, true, false, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

	// show logout every 10 secs, enabled with level 1 and output to stdout.
	s->setLogCycle(10, 0, stdout);

	// put spike times into spikes.dat
	s->setSpikeMonitor(g1, spikeCtrl);

	// Show basic statistics about g2
	s->setSpikeMonitor(g2, spikeCtrl);

	s->setSpikeMonitor(gin);

	//setup some baseline input
	PoissonRate in(N * 0.8);
	for (int i = 0; i < N * 0.8; i++) in.rates[i] = 1;
	s->setSpikeRate(gin, &in);


	//run the network interactively
	while (csc->execute) {
		while (csc->run) {
			s->runNetwork(1, 0);
		}
	}

	FILE* nid = fopen("network.dat","wb");
	s->writeNetwork(nid);
	fclose(nid);

	delete s;
	delete spikeCtrl;
	
	CLOSE_SOCKET(dataSocket);

	return 0;
}

int main() {
#if (WIN32 || WIN64)
	WSADATA wsaData;
#endif
    int iResult;

    SOCKET listenSocket = INVALID_SOCKET;
    SOCKET clientSocket = INVALID_SOCKET;

	SOCKADDR_IN clientAddr;
	
	ADDRINFO *result = NULL;
    ADDRINFO hints;

    int iSendResult;
    char recvBuf[DEFAULT_BUFLEN];
	char sendBuf[DEFAULT_BUFLEN];
    int bufLen = DEFAULT_BUFLEN;
	int numBytes;

	bool serverLoop = false;
	bool serviceThreadExe = false;
	
	// platform specific variable
#if (WIN32 || WIN64)
	HANDLE serviceThread = NULL;
	int clientAddrLen = sizeof(SOCKADDR_IN);
#else
	pthread_t serviceThread = 0;
	unsigned int clientAddrLen = sizeof(SOCKADDR_IN);
#endif

	CARLsimServiceConfig serviceConfig;
	serviceConfig.run = false;
	serviceConfig.execute = false;
	serviceConfig.display = false;
    
#if (WIN32 || WIN64)
    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed with error: %d\n", iResult);
        return 1;
    }
#endif

    //ZeroMemory(&hints, sizeof(hints));
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    iResult = getaddrinfo(NULL, DEFAULT_TCPIP_PORT, &hints, &result);
    if ( iResult != 0 ) {
        printf("getaddrinfo failed with error: %d\n", iResult);
        SOCKET_CLEAN_UP;
        return 1;
    }

    // Create a SOCKET for connecting to server
    listenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (listenSocket == INVALID_SOCKET) {
        printf("socket failed with error: %d\n", SOCKET_LAST_ERROR);
        freeaddrinfo(result);
        SOCKET_CLEAN_UP;
        return 1;
    }

    // Setup the TCP listening socket
    iResult = bind(listenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        printf("bind failed with error: %d\n", SOCKET_LAST_ERROR);
        freeaddrinfo(result);
        CLOSE_SOCKET(listenSocket);
        SOCKET_CLEAN_UP;
        return 1;
    }

    freeaddrinfo(result);
    
	iResult = listen(listenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        printf("listen failed with error: %d\n", SOCKET_LAST_ERROR);
        CLOSE_SOCKET(listenSocket);
        SOCKET_CLEAN_UP;
        return 1;
    }

    // Receive until the peer shuts down the connection
	serverLoop = true;
	do {
		// Accept a client socket
		if (clientSocket != INVALID_SOCKET) {
			CLOSE_SOCKET(clientSocket);
			clientSocket = INVALID_SOCKET;
		}

		printf("Waiting for connection...\n");
		clientSocket = accept(listenSocket, (SOCKADDR*)&clientAddr, &clientAddrLen);
		if (clientSocket == INVALID_SOCKET) {
			printf("accept failed with error: %d\n", SOCKET_LAST_ERROR);
			CLOSE_SOCKET(listenSocket);
			SOCKET_CLEAN_UP;
			return 1;
		}

		printf("Connection with %s is established!\n", inet_ntoa(clientAddr.sin_addr));

		// No longer need server socket
		//CLOSE_SOCKET(listenSocket);

		// Setup client address for udp socket
		//clientAddr.sin_addr // the same as vaule got from accept()
		clientAddr.sin_family = AF_INET;
		clientAddr.sin_port = htons(DEFAULT_UDP_PORT);

		serviceConfig.clientAddr = clientAddr;
		
		do {

			numBytes = recv(clientSocket, recvBuf, bufLen, 0);
			if (numBytes > 0) {
				printf("Bytes received: %d[%x]\n", numBytes, recvBuf[0]);

				// Process client requests
				sendBuf[0] = SERVER_RES_ACCEPT;
				iSendResult = send(clientSocket, sendBuf, 1 /* 1 byte */, 0);
				if (iSendResult == SOCKET_ERROR) {
					printf("send failed with error: %d\n", SOCKET_LAST_ERROR);
					//CLOSE_SOCKET(clientSocket);
					//SOCKET_CLEAN_UP;
					//return 1;
				}
				printf("Bytes sent: %d\n", iSendResult);

				//printf("%x\n", recvBuf[0]);
				switch (recvBuf[0]) {
					case CLIENT_REQ_START_SNN:
						//printf("run SNN\n");
						//runSNN(spikeCtrl);
						if (!serviceThreadExe) {
							serviceConfig.run = true;
							serviceConfig.execute = true;
							serviceConfig.display = false;
#if (WIN32 || WIN64)
							serviceThread = CreateThread(NULL, 0, service, (LPVOID)&serviceConfig, 0, NULL);
#else
							int ret = pthread_create(&serviceThread, NULL, service, (void*)&serviceConfig);
							printf("return of pthread_crate():%d [%ld]\n", ret, serviceThread);
#endif
							serviceThreadExe = true;
						} else {
							serviceConfig.run = true;
						}
						break;
					case CLIENT_REQ_STOP_SNN:
						if (serviceThreadExe) {
							serviceConfig.display = false;
							serviceConfig.run = false;
							serviceConfig.execute = false;
#if (WIN32 || WIN64)
							WaitForSingleObject(serviceThread, INFINITE);
							CloseHandle(serviceThread);
#else
							pthread_join(serviceThread, NULL);
#endif
							//serviceThread = NULL;
							serviceThreadExe = false;
						}
						break;
					case CLIENT_REQ_PAUSE_SNN:
						if (serviceThreadExe)
							serviceConfig.run = false;
						break;
					case CLIENT_REQ_START_SEND_SPIKE:
						//if (spikeCtrl == NULL)
						//	spikeCtrl = new SpikeController(dataSocket, clientAddr);
						if (serviceThreadExe)
							serviceConfig.display = true;
						break;
					case CLIENT_REQ_STOP_SEND_SPIKE:
						if (serviceThreadExe)
							serviceConfig.display = false;
						break;
					case CLIENT_REQ_SERVER_SHUTDOWN:
						serverLoop = false;
						break;
					default:
						// do nothing
						break;
				}

			}
			else if (numBytes == 0)
				printf("Connection closing...\n");
			else  {
				printf("Client closed the connection, error: %d\n", SOCKET_LAST_ERROR);
				//CLOSE_SOCKET(clientSocket);
				//SOCKET_CLEAN_UP;
				//return 1;
			}

		} while (numBytes > 0);
	} while (serverLoop);


    // shutdown the connection since we're done
    iResult = shutdown(clientSocket, SHUT_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", SOCKET_LAST_ERROR);
        CLOSE_SOCKET(clientSocket);
        SOCKET_CLEAN_UP;
        return 1;
    }

	if (listenSocket != INVALID_SOCKET)
		CLOSE_SOCKET(listenSocket);

	// clean up
	CLOSE_SOCKET(clientSocket);
	SOCKET_CLEAN_UP;

	return 0;
}
