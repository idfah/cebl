#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <scsi/sg.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <unistd.h>

#include "libmindset24r.h"


/*** debug and warings ***/

#define success 0
#define failure -1

#define debug_warn(...) debug_warn_w(__FILE__, __LINE__, __VA_ARGS__)
#define debug_warn_errno(...) debug_warn_errno_w(__FILE__, __LINE__, __VA_ARGS__)

unsigned debug = 0;

void ms_SetDebug(unsigned level)
{
  debug = level;
}

void debug_print(unsigned level, char *format, ...)
{
  va_list arglist;
  va_start(arglist, format);

  if (debug >= level) {
    vfprintf(stdout, format, arglist);
    fprintf(stdout, ".\n");
  }

  va_end(arglist);
}

void debug_warn_w(char *file, unsigned line, char *format, ...)
{
  va_list arglist;
  va_start(arglist, format);

  if (debug >= 0) {
    fprintf(stderr, "%s:%d: ", file, line);
    vfprintf(stderr, format, arglist);
    fprintf(stderr, ".\n");
  }

  va_end(arglist);
}

void debug_warn_errno_w(char *file, unsigned line, char *format, ...)
{
  va_list arglist;
  va_start(arglist, format);

  if (debug >= 0) {
    fprintf(stderr, "%s:%d: ", file, line);
    vfprintf(stderr, format, arglist);
    fprintf(stderr, ": %s.\n", strerror(errno));
  }

  va_end(arglist);
}

/*** driver ***/

int sg_fd;
sg_io_hdr_t io_hdr;

#define senseBufferLen 32
unsigned char senseBuffer[senseBufferLen];

/* see notes for "Get Mindset Status" in "Mindset24 SCSI Command Set" document */
static const int DeviceBufferBegin = 0x2000;
static const int DeviceBufferEnd   = 0x9DFF;

unsigned char *dataBytes = NULL;
int nDataBytes;

int ms_BytesToVolts(unsigned char *bytes, int nBytes, double *volts, int nVolts)
{
    int nSamples, chan, sample, v, b;

    debug_print(1, "Converting nBytes=%d to nVolts=%d", nBytes, nVolts);

    nSamples = nBytes / 24 / 2;

    if (nSamples > nVolts) {
        debug_warn("Received %d bytes, but only double array of length %d", nBytes, nVolts);
        return 0;
    }

    chan = 0;

    b = 0;
    v = 0;
    for (sample = 0; sample < nSamples; sample++)
        for (chan = 0; chan < 24; chan++) {
            volts[v] = ((bytes[b] << 8 | bytes[b+1]) - 0x7000) / 363.63;
            b += 2;
            v++;
        }

    return nSamples;
}  

int ms_SendCommand(Command command, int arg, unsigned char *reply, int nReply)
{
    debug_print(1, "Sending command %d", command);
    debug_print(1, "Value of arg is %d", arg);

    if(sg_fd < 0) {
        debug_warn("SCSI device file descriptor not set");
        return failure;
    }

    static unsigned char commandBlock[6];
    memset(&commandBlock, 0, 6);

    memset(&io_hdr, 0, sizeof(sg_io_hdr_t));
    io_hdr.interface_id = 'S';
    io_hdr.iovec_count = 0;
    io_hdr.mx_sb_len = senseBufferLen;
    io_hdr.cmd_len = sizeof(commandBlock);
    io_hdr.dxfer_len = nReply;
    io_hdr.dxferp = reply;
    io_hdr.cmdp = commandBlock;
    io_hdr.sbp = senseBuffer;
    io_hdr.timeout = 5000;      /* 20000 millisecs == 20 seconds */
    /* io_hdr.flags = 0; */     /* take defaults: indirect IO, etc */
    /* io_hdr.pack_id = 0; */
    /* io_hdr.usr_ptr = NULL; */

    switch(command) {
        case INQUIRE:
            commandBlock[0] = 0x12;
            commandBlock[4] = nReply;
            io_hdr.dxfer_direction = SG_DXFER_FROM_DEV;
            break;

        case READDATA:
            commandBlock[0] = 0xc0;
            commandBlock[1] = 0x04;
            commandBlock[2] = arg;
            io_hdr.dxfer_direction = SG_DXFER_FROM_DEV;
            break;

        case READSTATUS:
            commandBlock[0] = 0xc0;
            commandBlock[1] = 0x06;
            commandBlock[4] = 0x1e;
            io_hdr.dxfer_direction = SG_DXFER_FROM_DEV;
            break;

        case READY:
            commandBlock[0] = 0x00;
            io_hdr.dxfer_direction = SG_DXFER_FROM_DEV;
            break;

        case SETSAMPLERATE:
            commandBlock[0] = 0xc0;
            commandBlock[1] = 0x03;
            commandBlock[2] = arg; // Sample rate
            io_hdr.dxfer_direction = SG_DXFER_TO_DEV;
            break;

        case SETBLOCKSIZE:
            commandBlock[0] = 0xc0;
            commandBlock[1] = 0x05;
            commandBlock[2] = arg; // Block size
            io_hdr.dxfer_direction = SG_DXFER_TO_DEV;
            break;

        default:
            debug_warn("Invalid command %d", command);
            return failure;
    }

    if (debug > 0)
        ms_Dump(commandBlock, 6, "Starting command: ");

    if (ioctl(sg_fd, SG_IO, &io_hdr) < 0) {
        debug_warn_errno("ioctl failed\n");
        return failure;
    }

    if ( (senseBuffer[2] == 0x09)  &&
         (senseBuffer[13] == 0x00) )
        debug_warn("Overflow");

    if ((io_hdr.info & SG_INFO_OK_MASK) != SG_INFO_OK) {
        //debug_warn("(io_hdr.info & SG_INFO_OK_MASK) != SG_INFO_OK");
        debug_warn("io_hdr.info=0x%x", io_hdr.info);

        if (io_hdr.sb_len_wr > 0)
            ms_Dump(senseBuffer, io_hdr.sb_len_wr, "SCSI sense data: ");
        if (io_hdr.masked_status)
            debug_warn("SCSI status=0x%x", io_hdr.status);
        if (io_hdr.host_status)
            debug_warn("SCSI host_status=0x%x", io_hdr.host_status);
        if (io_hdr.driver_status)
            debug_warn("driver_status=0x%x", io_hdr.driver_status);
        else {
            ms_Dump(reply, nReply, "SCSI reply");
            debug_warn("SCSI command duration=%u millisecs, resid=%d",
                        io_hdr.duration, io_hdr.resid);
        }
        return failure;
    }

    if (command == READY)
        return (io_hdr.masked_status == 0x0);
    else
        return success;
}

int ms_Open(const char *dev)
{
    debug_print(1, "Opening %s", dev);

    sg_fd = open(dev, O_RDWR);

    if (sg_fd < 0) {
        debug_warn_errno("Failed to open %s", dev);
        return failure;
    }

    debug_print(1, "sg_fd open returned %d", sg_fd);

    return success;
}

int ms_Close()
{
    int status = success;

    debug_print(1, "Setting sample rate to zero");
    if (ms_SendCommand(SETSAMPLERATE, 0, NULL, 0) == failure) {
        debug_warn("Failed to set sample rate to zero.");
        status = failure;
    }

    debug_print(1, "Closing sg_fd");
    if (close(sg_fd) < 0) {
        debug_warn_errno("Failed to close sg_fd");
        status = failure;
    }

    if (dataBytes != NULL) {
        debug_print(1, "Freeing dataBytes");
        free(dataBytes);
        dataBytes = NULL;
    }

    return status;
}

int ms_Ready()
{
    debug_print(1, "Sending Test Unit Ready");

    return ms_SendCommand(READY, 0, NULL, 0);
}

int ms_ReadStatus(SampleRate *sampleRate, BlockSize *blockSize, int *numDataBytes)
{
    unsigned char reply[32];
    unsigned short shortHead;
    unsigned short shortTail;
    int nBytes;

    debug_print(1, "Sending Read Status");

    if (ms_SendCommand(READSTATUS, 0, reply, sizeof(reply)) == failure) {
        debug_warn("Failed to send READSTATUS");
        return failure;
    }

    debug_print(1, "Result is sampleRate %d, blockSize %d",
                (SampleRate)reply[0], (BlockSize)reply[5]);

    if (sampleRate != NULL)
        *sampleRate = (SampleRate)reply[0];

    if (blockSize != NULL)
        *blockSize = (BlockSize)reply[5];

    shortHead = reply[1] << 8 | reply[2];
    shortTail = reply[3] << 8 | reply[4];

    if (shortHead <= shortTail)
        nBytes = shortTail - shortHead;
    else
        nBytes = shortTail - DeviceBufferBegin + DeviceBufferEnd - shortHead;

    if (numDataBytes != NULL) 
        *numDataBytes = nBytes;

    debug_print(1, " head %x  tail %x  devicebuffer begin %x  "
                "end %x nBytesAvaliable %d", shortHead, shortTail,
                DeviceBufferBegin, DeviceBufferEnd, nBytes);

    return success;
}

int ms_SetBlockSize(BlockSize blockSize)
{
    if (dataBytes != NULL) {
        debug_print(1, "Freeing dataBytes");
        free(dataBytes);
        dataBytes = NULL;
    }

    debug_print(1, "Allocating dataBytes");
    nDataBytes = ms_ActualBlockSize(blockSize);
    dataBytes = (unsigned char *)malloc(nDataBytes);

    if (dataBytes == NULL) {
        debug_warn_errno("Failed to malloc dataBytes");
        return failure;
    }

    debug_print(1, "Setting block size");
    return ms_SendCommand(SETBLOCKSIZE, blockSize, NULL, 0);
}

int ms_ActualBlockSize(BlockSize blockSize)
{
    debug_print(1, "ms_ActualBlockSize received %d\n", blockSize);

    switch (blockSize) {
        case BLOCKSIZE96:
            return 96;

        case BLOCKSIZE192:
            return 192;

        case BLOCKSIZE384:
            return 384;

        case BLOCKSIZE768:
            return 768;

        default: 
            debug_warn("Invalid blockSize %d", blockSize);
            return failure;
    }

    return failure;
}

int ms_SetSampleRate(SampleRate sampleRate)
{
    debug_print(1, "Setting sample rate to %d", sampleRate);
    return ms_SendCommand(SETSAMPLERATE, sampleRate, NULL, 0);
}

int ms_ActualSampleRate(SampleRate sampleRate)
{
    debug_print(1, "ms_ActualSampleRate received %d\n", sampleRate);

    switch (sampleRate) {
        case SAMPLERATE0:
            return 0;

        case SAMPLERATE1024:
            return 1024;

        case SAMPLERATE512:
            return 512;

        case SAMPLERATE256:
            return 256;

        case SAMPLERATE128:
            return 128;

        case SAMPLERATE64:
            return 64;

        default: 
            debug_warn("Invalid sampleRate %d", sampleRate);
            return failure;
    }

    return failure;
}

int ms_ReadAllDataBlocks(double *data, int nSamples)
{
    BlockSize blockSize;
    int bytesAvailable = 0;
    int actualBlockSize;

    debug_print(1, "Reading data");

    if (ms_ReadStatus(NULL, &blockSize, &bytesAvailable) == failure) {
        debug_warn("Failed to read status");
        return failure;
    }

    actualBlockSize = ms_ActualBlockSize(blockSize);
    if (actualBlockSize == failure) {
        debug_warn("Failed to find actualBlockSize");
        return failure;
    }

    debug_print(1, "Bytes available=%d, actualBlockSize=%d", bytesAvailable, actualBlockSize);

    /* Wait until enough bytes available to fill one block */
    while (bytesAvailable < actualBlockSize) {
        if (ms_ReadStatus(NULL, &blockSize, &bytesAvailable) == failure) {
            debug_warn("Failed to read status");
            return failure;
        }

        debug_print(1, " Bytes available=%d, actualBlockSize=%d, need more bytes",
                    bytesAvailable, actualBlockSize);
    }

    debug_print(1, "Bytes available=%d, reading data now",
                bytesAvailable);

    unsigned char *output;
    int nBlocks = bytesAvailable / actualBlockSize;
    int nOutput = nBlocks * actualBlockSize;

    output = (unsigned char *)malloc(nOutput * sizeof(char));

    if (ms_SendCommand(READDATA, nBlocks, output, nOutput) == failure) {
        debug_warn("Failed to send READDATA command");
        return failure;
    }

    nSamples = ms_BytesToVolts(output, nOutput, data, nSamples);

    free(output);

    return nSamples;
}

int ms_ReadNextDataBlock(double *data, int nSamples)
{
    BlockSize blockSize;
    int bytesAvailable = 0;
    int actualBlockSize;

    debug_print(1, "Reading data");

    if (ms_ReadStatus(NULL, &blockSize, &bytesAvailable) == failure) {
        debug_warn("Failed to read status");
        return failure;
    }

    actualBlockSize = ms_ActualBlockSize(blockSize);
    if (actualBlockSize == failure) {
        debug_warn("Failed to find actualBlockSize");
        return failure;
    }

    debug_print(1, "Bytes available=%d, actualBlockSize=%d", bytesAvailable, actualBlockSize);

    /* Wait until enough bytes available to fill one block */
    while (bytesAvailable < actualBlockSize) {
        if (ms_ReadStatus(NULL, &blockSize, &bytesAvailable) == failure) {
            debug_warn("Failed to read status");
            return failure;
        }

        debug_print(1, " Bytes available=%d, actualBlockSize=%d, need more bytes",
                    bytesAvailable, actualBlockSize);
    }

    debug_print(1, "Bytes available=%d, reading data now",
                bytesAvailable);

    if (ms_SendCommand(READDATA, 1, dataBytes, nDataBytes) == failure) {
        debug_warn("Failed to send READDATA command");
        return failure;
    }

    return ms_BytesToVolts(dataBytes, nDataBytes, data, nSamples);
}

void ms_Dump(void *buffer, int count, char *str)
{
    int i, j;
    unsigned char *c = (unsigned char *)buffer;

    printf("%s\n", str);
    printf("count: %d.\n", count);

    for (i = 0; i < count; i += 16) {
        printf(" %04x: ",i);
        for (j = i; j < i+16; j++)
            if (j < count)
                printf("%02x ", c[j]);
            else
                printf("   ");
        for (j = i; j < i+16; j++)
            if (j < count)
                putchar(isprint(c[j]) ? c[j] : '.');
            else
                putchar(' ');
        putchar('\n');
    }
}
