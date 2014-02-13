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

#define N    1000

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

typedef struct group_data_t {
	unsigned int time;
	unsigned int grpId;
	float buf[100];
} GroupData;

class GroupController: public GroupMonitor {
private:
	SOCKET dataSocket;
	SOCKADDR_IN clientAddr;
	//float buf[BUF_LEN];
	GroupData grpData;
	//int bufPos;

public:
	GroupController(SOCKET ds, SOCKADDR_IN cd) {
		dataSocket = ds;
		clientAddr = cd;
		//bufPos = 0;
	}

	void update(CpuSNN* s, int grpId, float* daBuffer, int n) {
		// prepare group data
		grpData.time = 0xFFFFFFFF;
		grpData.grpId = grpId << 24;
		for (int i = 0; i < 100 /* n is 100 currently */; i++)
			grpData.buf[i] = daBuffer[i];

		int numByteSent = sendto(dataSocket, (char*)&grpData, 2 * sizeof(unsigned int) + 100 * sizeof(float), NULL, (SOCKADDR*)&clientAddr, sizeof(SOCKADDR_IN));
		printf("send out %d bytes udp data\n", numByteSent);
	}
};

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
				buf[bufPos + 1] = gId | id;
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
		if (bufPos > 0) {
			numByteSent = sendto(dataSocket, (char*)buf, bufPos * sizeof(unsigned int), NULL, (SOCKADDR*)&clientAddr, sizeof(SOCKADDR_IN));
			printf("send out %d bytes udp data on port %d\n", numByteSent, ntohs(clientAddr.sin_port));
		}

		bufPos = 0;
		//buf[0] = currentTimeSlice++;
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
	GroupController* groupCtrl = new GroupController(dataSocket, csc->clientAddr);

	int pfc, sen_cs, sen_us, ic_cs, ic_us, str, da;
	int pfc_input, sen_cs_input, sen_us_input;
	
	// create a spiking neural network
	s = new CpuSNN("global", GPU_MODE);

	//daController = new DopamineController(stdpLog);

	pfc = s->createGroup("PFC_Ex", 1000, EXCITATORY_NEURON);
	s->setNeuronParameters(pfc, 0.02f, 0.2f, -65.0f, 8.0f);

	//int g2 = s->createGroup("inhib", N * 0.2, INHIBITORY_NEURON);
	//s->setNeuronParameters(g2, 0.1f,  0.2f, -65.0f, 2.0f);

	// sensory neurons
	sen_cs = s->createGroup("Sensory_CS", 500, EXCITATORY_NEURON);
	s->setNeuronParameters(sen_cs, 0.02f, 0.2f, -65.0f, 8.0f);

	sen_us = s->createGroup("Sensory_US", 500, EXCITATORY_NEURON);
	s->setNeuronParameters(sen_us, 0.02f, 0.2f, -65.0f, 8.0f);

    // 200 striatum neurons
	str = s->createGroup("Stritum", 400, INHIBITORY_NEURON);
	s->setNeuronParameters(str, 0.02f, 0.2f, -65.0f, 8.0f);
	
	// ic neurons
	ic_cs = s->createGroup("Insular_CS", 200, EXCITATORY_NEURON);
	s->setNeuronParameters(ic_cs, 0.02f, 0.2f, -65.0f, 8.0f);

	ic_us = s->createGroup("Insular_US", 200, EXCITATORY_NEURON);
	s->setNeuronParameters(ic_us, 0.02f, 0.2f, -65.0f, 8.0f);
	
	// 100 dopaminergeic neurons
	da = s->createGroup("Dopaminergic Area", 50, DOPAMINERGIC_NEURON);
	s->setNeuronParameters(da, 0.02f, 0.2f, -65.0f, 8.0f);

	// stimulus 
	pfc_input = s->createSpikeGeneratorGroup("PFC input", 1000, EXCITATORY_NEURON);
	sen_cs_input = s->createSpikeGeneratorGroup("Sensory_CS input", 500, EXCITATORY_NEURON);
	sen_us_input = s->createSpikeGeneratorGroup("Sensory_US input", 500, EXCITATORY_NEURON);


	s->setWeightUpdateParameter(_10MS, 100);

	// make random connections with 10% probability
	//s->connect(g2, g1, "random", -4.0f/100, -4.0f/100, 0.1f, 1, 1, SYN_FIXED);
	// make random connections with 10% probability, and random delays between 1 and 20
	//s->connect(g1, g2, "random", 5.0f/100, 10.0f/100, 0.1f,  1, 20, SYN_PLASTIC);
	
	s->connect(pfc, str, "random", 2.8f/100, 10.0f/100, 0.04f, 1, 10, SYN_PLASTIC);

	s->connect(sen_cs, ic_cs, "random", 6.0f/100, 10.0f/100, 0.04f, 1, 10, SYN_PLASTIC);
	s->connect(sen_us, ic_us, "random", 0.5f/100, 10.0f/100, 0.04f, 1, 10, SYN_PLASTIC);

	s->connect(str, da, "random", -2.0f/100, -2.0f/100, 0.08f, 10, 10, SYN_FIXED);

	s->connect(ic_cs, da, "random", 3.6f/100, 3.6f/100, 0.08f, 10, 10, SYN_FIXED);
	s->connect(ic_us, da, "random", 3.6f/100, 3.6f/100, 0.08f, 10, 10, SYN_FIXED);

	// 5% probability of connection
	// Dummy synaptic weights. Dopaminergic neurons only release dopamine to the target area in the current model.
	s->connect(da, str, "random", 0.0, 0.0, 0.02f, 1, 20, SYN_FIXED);
	s->connect(da, ic_cs, "random", 0.0, 0.0, 0.05f, 1, 20, SYN_FIXED);
	s->connect(da, ic_us, "random", 0.0, 0.0, 0.05f, 1, 20, SYN_FIXED);

	// input connection
	s->connect(pfc_input, pfc, "one-to-one", 20.0f/100, 20.0f/100, 1.0f,  1, 1, SYN_FIXED);
	s->connect(sen_cs_input, sen_cs, "one-to-one", 20.0f/100, 20.0f/100, 1.0f, 1, 1, SYN_FIXED);
	s->connect(sen_us_input, sen_us, "one-to-one", 20.0f/100, 20.0f/100, 1.0f, 1, 1, SYN_FIXED);

	float COND_tAMPA = 5.0, COND_tNMDA = 150.0, COND_tGABAa = 6.0, COND_tGABAb = 150.0;
	s->setConductances(ALL, true, COND_tAMPA, COND_tNMDA, COND_tGABAa, COND_tGABAb);

	// here we define and set the properties of the STDP. 
	float ALPHA_LTP = 0.10f/100, TAU_LTP = 20.0f, ALPHA_LTD = 0.08f/100, TAU_LTD = 40.0f;	
	s->setSTDP(str, true, true, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
	s->setSTDP(ic_cs, true, true, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
	s->setSTDP(ic_us, true, true, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

	// show logout every 10 secs, enabled with level 1 and output to stdout.
	s->setLogCycle(10, 3, stdout);

	// put spike times into spikes.dat
	//s->setSpikeMonitor(g1,"spikes.dat");
	s->setSpikeMonitor(pfc, spikeCtrl);
	s->setSpikeMonitor(sen_cs, spikeCtrl);
	s->setSpikeMonitor(sen_us, spikeCtrl);
	s->setSpikeMonitor(ic_cs, spikeCtrl);
	s->setSpikeMonitor(ic_us, spikeCtrl);
	s->setSpikeMonitor(str, spikeCtrl);
	s->setSpikeMonitor(da, spikeCtrl);

	s->setGroupMonitor(str, groupCtrl);
	s->setGroupMonitor(ic_cs, groupCtrl);
	s->setGroupMonitor(ic_us, groupCtrl);

	//setup random thalamic noise
	PoissonRate pfc_input_rate(1000);
	for (int i = 0; i < 1000; i++)
		pfc_input_rate.rates[i] = 0.6;
	s->setSpikeRate(pfc_input, &pfc_input_rate);

	PoissonRate sen_cs_input_rate(500);
	for (int i = 0; i < 500; i++)
		sen_cs_input_rate.rates[i] = 0.6;
	s->setSpikeRate(sen_cs_input, &sen_cs_input_rate);
	
	PoissonRate sen_us_input_rate(500);
	for (int i = 0; i < 500; i++)
		sen_us_input_rate.rates[i] = 0.6;
	s->setSpikeRate(sen_us_input, &sen_us_input_rate);

	//s->setSpikeGenerator(pfc_input, (SpikeGenerator*)spikeCtrl);
	//s->setSpikeGenerator(sen_cs_input, (SpikeGenerator*)spikeCtrl);
	//s->setSpikeGenerator(sen_us_input, (SpikeGenerator*)spikeCtrl);

	//run for 60 seconds
	while (csc->execute) {
		// run the established network for a duration of 1 (sec)  and 0 (millisecond), in CPU_MODE
		while (csc->run) {
			s->runNetwork(1, 0);
		}
	}

	FILE* nid = fopen("network.dat","wb");
	s->writeNetwork(nid);
	fclose(nid);

	delete s;

	delete spikeCtrl;
	delete groupCtrl;
	
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
