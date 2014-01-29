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

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <stdio.h>

#include "snn.h"
#include "server_client.h"

#define N    1000

#define DEFAULT_BUFLEN 128
#define DEFAULT_TCPIP_PORT "27016"
#define DEFAULT_UDP_PORT 27000

#define BUF_LEN 128

#pragma comment(lib, "Ws2_32.lib")

class SpikeController: public SpikeMonitor, SpikeGenerator {
private:
	SOCKET dataSocket;
	SOCKADDR_IN clientAddr;
	unsigned int buf[BUF_LEN];
	//unsigned int currentTimeSlice;
	int bufPos;

public:
	SpikeController(SOCKET ds, SOCKADDR_IN cd) {
		dataSocket = ds;
		clientAddr = cd;
		bufPos = 0;
		//for (int i = 0; i < 16; i++)
		//	buf[i] = i;

		//currentTimeSlice = 0;

	}

	// nextSpikeTime is called every one second (simulation time)
	unsigned int nextSpikeTime(CpuSNN* s, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		//currentTimeSlice = currentTime;
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
				//int cnt = fwrite(&time, sizeof(int), 1, fid);
				//assert(cnt != 0);
				//cnt = fwrite(&id, sizeof(int), 1, fid);
				//assert(cnt != 0);

				buf[bufPos] = time;
				buf[bufPos + 1] = id | gId;
				bufPos += 2;

				// send out data if buffer is full
				if (bufPos >= BUF_LEN) {
					int numByteSent = sendto(dataSocket, (char*)buf, BUF_LEN * sizeof(unsigned int), NULL, (SOCKADDR*)&clientAddr, sizeof(SOCKADDR_IN));
					printf("send out %d bytes udp data on port %d\n", numByteSent, ntohs(clientAddr.sin_port));

					bufPos = 0;
				}
			}
		}

		// send out the rest of data
		numByteSent = sendto(dataSocket, (char*)buf, bufPos * sizeof(unsigned int), NULL, (SOCKADDR*)&clientAddr, sizeof(SOCKADDR_IN));
		printf("send out %d bytes udp data on port %d\n", numByteSent, ntohs(clientAddr.sin_port));

		bufPos = 0;
		//buf[0] = currentTimeSlice++;
	}
};

void runSNN(SpikeController* spikeCtrl) {
	CpuSNN* s;

	// Create Spiking Neural Network
	s = new CpuSNN("global", GPU_MODE);

	int g1 = s->createGroup("excit", N * 0.8, EXCITATORY_NEURON);
	s->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int g2 = s->createGroup("inhib", N * 0.2, INHIBITORY_NEURON);
	s->setNeuronParameters(g2, 0.1f,  0.2f, -65.0f, 2.0f);

	int gin = s->createSpikeGeneratorGroup("input", N * 0.8, EXCITATORY_NEURON);

	s->setWeightUpdateParameter(_1000MS, 100);
	
	// make random connections with 10% probability
	s->connect(g2, g1, "random", -2.0f/100, -2.0f/100, 0.1f, 1, 1, SYN_FIXED);
	// make random connections with 10% probability, and random delays between 1 and 20
	s->connect(g1, g2, "random", +2.5f/100, 5.0f/100, 0.1f,  1, 20, SYN_PLASTIC);
	s->connect(g1, g1, "random", +4.0f/100, 10.0f/100, 0.1f,  1, 20, SYN_PLASTIC);

	// 5% probability of connection
	s->connect(gin, g1, "one-to-one", +20.0f/100, 20.0f/100, 1.0f,  1, 20, SYN_FIXED);

	float COND_tAMPA = 5.0, COND_tNMDA = 150.0, COND_tGABAa = 6.0, COND_tGABAb = 150.0;
	s->setConductances(ALL, true, COND_tAMPA, COND_tNMDA, COND_tGABAa, COND_tGABAb);

	// here we define and set the properties of the STDP. 
	float ALPHA_LTP = 0.10f/100, TAU_LTP = 20.0f, ALPHA_LTD = 0.08f/100, TAU_LTD = 40.0f;	
	s->setSTDP(g1, true, false, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
	s->setSTDP(g2, true, false, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

	// show logout every 10 secs, enabled with level 1 and output to stdout.
	s->setLogCycle(10, 3, stdout);

	// put spike times into spikes.dat
	s->setSpikeMonitor(g1, spikeCtrl);

	// Show basic statistics about g2
	s->setSpikeMonitor(g2);

	s->setSpikeMonitor(gin);

	//setup some baseline input
	PoissonRate in(N * 0.8);
	for (int i = 0; i < N * 0.8; i++) in.rates[i] = 1;
	s->setSpikeRate(gin,&in);

	//run for 60 seconds
	for(int i=0; i < 20; i++) {
		// run the established network for a duration of 1 (sec)  and 0 (millisecond), in CPU_MODE
		s->runNetwork(1, 0);
	}

	FILE* nid = fopen("network.dat","wb");
	s->writeNetwork(nid);
	fclose(nid);

	delete s;
}

int main() {
	WSADATA wsaData;
    int iResult;

    SOCKET listenSocket = INVALID_SOCKET;
    SOCKET clientSocket = INVALID_SOCKET;
	SOCKET dataSocket = INVALID_SOCKET;

	SOCKADDR_IN clientAddr;
	
	ADDRINFO *result = NULL;
    ADDRINFO hints;

    int iSendResult;
    char recvBuf[DEFAULT_BUFLEN];
	char sendBuf[DEFAULT_BUFLEN];
    int bufLen = DEFAULT_BUFLEN;
	int numBytes;
	int clientAddrLen = sizeof(SOCKADDR_IN);

	bool serverLoop = false;

	SpikeController* spikeCtrl = NULL;
    
    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed with error: %d\n", iResult);
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    iResult = getaddrinfo(NULL, DEFAULT_TCPIP_PORT, &hints, &result);
    if ( iResult != 0 ) {
        printf("getaddrinfo failed with error: %d\n", iResult);
        WSACleanup();
        return 1;
    }

    // Create a SOCKET for connecting to server
    listenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (listenSocket == INVALID_SOCKET) {
        printf("socket failed with error: %ld\n", WSAGetLastError());
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }

    // Setup the TCP listening socket
    iResult = bind(listenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        printf("bind failed with error: %d\n", WSAGetLastError());
        freeaddrinfo(result);
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    freeaddrinfo(result);
    
	iResult = listen(listenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        printf("listen failed with error: %d\n", WSAGetLastError());
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    dataSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	
    // Receive until the peer shuts down the connection
	serverLoop = true;
	do {
		// Accept a client socket
		if (clientSocket != INVALID_SOCKET) {
			closesocket(clientSocket);
			clientSocket = INVALID_SOCKET;
		}

		printf("Waiting for connection...\n");
		clientSocket = accept(listenSocket, (SOCKADDR*)&clientAddr, &clientAddrLen);
		if (clientSocket == INVALID_SOCKET) {
			printf("accept failed with error: %d\n", WSAGetLastError());
			closesocket(listenSocket);
			WSACleanup();
			return 1;
		}

		printf("Connection with %s is established!\n", inet_ntoa(clientAddr.sin_addr));

		// No longer need server socket
		//closesocket(listenSocket);

		// Setup client address for udp socket
		//clientAddr.sin_addr // the same as vaule got from accept()
		clientAddr.sin_family = AF_INET;
		clientAddr.sin_port = htons(DEFAULT_UDP_PORT);
		
		do {

			numBytes = recv(clientSocket, recvBuf, bufLen, 0);
			if (numBytes > 0) {
				printf("Bytes received: %d\n", numBytes);

				// Process client requests
				sendBuf[0] = SERVER_RES_ACCEPT;
				iSendResult = send(clientSocket, sendBuf, 1 /* 1 byte */, 0);
				if (iSendResult == SOCKET_ERROR) {
					printf("send failed with error: %d\n", WSAGetLastError());
					closesocket(clientSocket);
					WSACleanup();
					return 1;
				}
				printf("Bytes sent: %d\n", iSendResult);

				//printf("%x\n", recvBuf[0]);
				switch (recvBuf[0]) {
					case CLIENT_REQ_START_SNN:
						printf("run SNN\n");
						runSNN(spikeCtrl);
						break;
					case CLIENT_REQ_START_SEND_SPIKE:
						if (spikeCtrl == NULL)
							spikeCtrl = new SpikeController(dataSocket, clientAddr);
						break;
					case CLIENT_REQ_STOP_SEND_SPIKE:
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
				printf("Client closed the connection, error: %d\n", WSAGetLastError());
				//closesocket(clientSocket);
				//WSACleanup();
				//return 1;
			}

		} while (numBytes > 0);
	} while (serverLoop);


    // shutdown the connection since we're done
    iResult = shutdown(clientSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
        closesocket(clientSocket);
        WSACleanup();
        return 1;
    }

	if (spikeCtrl != NULL)
		delete spikeCtrl;
	
	if (dataSocket != INVALID_SOCKET)
		closesocket(dataSocket);
	if (listenSocket != INVALID_SOCKET)
		closesocket(listenSocket);

	// clean up
	closesocket(clientSocket);
	WSACleanup();

	return 0;
}
