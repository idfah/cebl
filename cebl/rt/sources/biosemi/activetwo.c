#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bsif.h"

#include "activetwo.h"


//** private variables **//

static int logging = 0;
static int debugging = 0;
static int syncPointer = 1;

static int stride_kb = 0;
static int stride_ms = 0;

static bool isOpen = false;
static bool isRunning = false;
static void *usbHandle = NULL;

static char controlBuffer[64];

static long head = 0, seam = 0;
static unsigned dataBufferSize = 32768 * 1024;
static unsigned char *dataBuffer = NULL;


//** public variables **//

bs_speedMode_t bs_speedMode = BS_SPEED_MODE_4;
unsigned bs_nChan = 2+256+24;
unsigned bs_sampRate = 2048;
unsigned bs_scansPerPoll = 16;

//** public functions **//

int bs_setSpeedMode(bs_speedMode_t speedMode)
{
    bs_speedMode = speedMode;

    if (bs_speedMode == BS_SPEED_MODE_4) {
        bs_nChan = 256;
        bs_sampRate = 2048;
    }
    else if (bs_speedMode == BS_SPEED_MODE_5) {
        bs_nChan = 128;
        bs_sampRate = 4096;
    }
    else if (bs_speedMode == BS_SPEED_MODE_6) {
        bs_nChan = 64;
        bs_sampRate = 8192;
    }
    else if (bs_speedMode == BS_SPEED_MODE_7) {
        bs_nChan = 32;
        bs_sampRate = 16384;
    }
    else {
        fprintf(stderr, "Invalid speed mode.\n");
        return 1;
    }

    bs_nChan += 2+24;

    return 0;
}

int bs_setScansPerPoll(unsigned scans)
{
    bs_scansPerPoll = scans;

    return 0;
}

int bs_open()
{
    if (isOpen) {
        fprintf(stderr, "Driver is already open.\n");
        return 1;
    }

    else if ((usbHandle = BSIF_OPEN_DRIVER()) == NULL) {
        fprintf(stderr, "Failed to open USB device.\n");
        isOpen = false;
        return 1;
    }

    else {
        BSIF_SET_LOG(logging);
        BSIF_SET_DEBUG(debugging);

        BSIF_SET_SYNC(syncPointer);
        BSIF_SET_STRIDE_KB(stride_kb);
        BSIF_SET_STRIDE_MS(stride_ms);

        usleep(500*1000); // probably not necessary XXX - idfah

        isOpen = true;
        return 0;
    }
}

int bs_close()
{
    if (isOpen)
        BSIF_CLOSE_DRIVER(usbHandle);
    else
        fprintf(stderr, "Driver is already closed.\n");

    usleep(500*1000); // probably not necessary XXX - idfah

    isOpen = false;
    return 0;
}

int bs_start()
{
    if (!isOpen) {
        fprintf(stderr, "Driver must be opened before starting.\n");
        return 1;
    }

    if (isRunning) {
        fprintf(stderr, "Acquisition is already running.\n");
        return 1;
    }

    head = seam = 0;

    // allocate memory for data buffer
    if ((dataBuffer = (unsigned char*)malloc(sizeof(char)*dataBufferSize)) == NULL) {
        fprintf(stderr, "Failed to allocate memory for dataBuffer.\n");
        isRunning = false;
        return 1;
    }

    // initialize ring buffer on dataBuffer
    memset(dataBuffer, 0, sizeof(char)*dataBufferSize);
    if (!BSIF_READ_MULTIPLE_SWEEPS(usbHandle, (char*)dataBuffer, dataBufferSize)) {
        fprintf(stderr, "Failed to initialize ring buffer.\n");
        isRunning = false;
        return 1;
    }
    usleep(500*1000); // prolly not necessary XXX - idfah

    // start handshake
    memset(controlBuffer, 0, sizeof(char)*64); // clear control buffer
    controlBuffer[0] = 0xFF; // set instruction in first byte
    if (!BSIF_USB_WRITE(usbHandle, controlBuffer)) {
        fprintf(stderr, "Failed to start handshake.\n");
        isRunning = false;
        return 1;
    }
    usleep(500*1000); // prolly not necessary XXX - idfah

    isRunning = true;
    return 0;
}

int bs_stop()
{
    if (!isRunning) {
        fprintf(stderr, "Acquisition is already stopped.\n");
        return 1;
    }

    if (!isOpen) {
        fprintf(stderr, "Driver must be opened before stopping.\n");
        return 1;
    }


    // stop handshake
    memset(controlBuffer, 0, sizeof(char)*64);
    if (!BSIF_USB_WRITE(usbHandle, controlBuffer)) {
        fprintf(stderr, "Failed to stop handshake.\n");
        isRunning = true;
        return 1;
    }

    usleep(500*1000); // probably not necessary XXX - idfah

    free(dataBuffer);
    dataBuffer = NULL;

    isRunning = false;
    return 0;
}

int bs_poll(double *pollBuffer)
//int bs_poll(char *pollBuffer)
{
    unsigned i, j;
    long bytesAvailable = 0;
    unsigned valuesPerPoll = bs_scansPerPoll * bs_nChan;
    unsigned bytesPerPoll = valuesPerPoll * 4;

    /*printf("A\n");
    fflush(stdout);*/

    while (true) {
        bytesAvailable = seam - head;
        /*printf("bytesAvailable: %d\n", bytesAvailable);
        printf("seam: %ld\n", seam);
        printf("head: %ld\n", head);
        printf("bytesPerPoll: %d\n", bytesPerPoll);
        fflush(stdout);*/

        if (bytesAvailable < 0)
            bytesAvailable += dataBufferSize;

        if (bytesAvailable >= dataBufferSize) {
            fprintf(stderr, "Poll buffer overflow.\n");
            return 1;
        }

        if (bytesAvailable > bytesPerPoll)
            break;

        if (!BSIF_READ_POINTER(usbHandle, &seam)) {
            fprintf(stderr, "Failed to get read pointer.\n");
            return 1;
        }
    }

    memset(pollBuffer, 0, sizeof(double)*(valuesPerPoll));
    for (i = 0, j = 0; i < bytesPerPoll; i += 4) {
        unsigned off = (head+i) % dataBufferSize;
        unsigned chan = j % bs_nChan;

        unsigned data = (dataBuffer[off+3] << 24 |
                         dataBuffer[off+2] << 16 |
                         dataBuffer[off+1] << 8 |
                         dataBuffer[off+0]);

        /*printf("%d %d %d %d %d\n",
                dataBuffer[off+3],
                dataBuffer[off+2],
                dataBuffer[off+1],
                dataBuffer[off], data);
        printf("%02X %02X %02X %02X %d\n", dataBuffer[off+3], dataBuffer[off+2], dataBuffer[off+1], dataBuffer[off], data);
        fflush(stdout);*/

        if ((chan == 0)) {//&& (data != 0xFFFFFF00)) {
            //printf("%02X %02X %02X %02X %d\n", dataBuffer[off+3], dataBuffer[off+2], dataBuffer[off+1], dataBuffer[off], data);
            if (data != 0xFFFFFF00) {
                fprintf(stderr, "Poll buffer not synchronized.\n");
                return 1;
            } /*else {
                printf("Sync.\n");
                fflush(stdout);
            }*/
        } 

        if (chan < 2)
            pollBuffer[j++] = (double) data;
        else
            pollBuffer[j++] = data / (32.0*256.0);
    }
    /*memset(pollBuffer, 0, sizeof(char)*(bytesPerPoll));
    for (i = 0, j = 0; i < bytesPerPoll; i++) {
        unsigned off = (head+i) % dataBufferSize;
        pollBuffer[j++] = dataBuffer[off];
    }*/
    //lastSeam = seam;
    head += bytesPerPoll;
    //head %= dataBufferSize;
    if (head >= dataBufferSize) {
        printf("rolling...\n");
        fflush(stdout);
        head -= dataBufferSize;
    }

    return 0;
}
