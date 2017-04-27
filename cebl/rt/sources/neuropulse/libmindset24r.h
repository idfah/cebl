#ifndef libmindset24_h
    #define libmindset24_h

    typedef enum {
        INQUIRE,
        READDATA,
        READSTATUS,
        READY,
        SETSAMPLERATE,
        SETBLOCKSIZE
    } Command;

    typedef enum {
        BLOCKSIZE96  = 1,
        BLOCKSIZE192 = 2,
        BLOCKSIZE384 = 4,
        BLOCKSIZE768 = 8
    } BlockSize;

    typedef enum {
        SAMPLERATE0    = 0,
        SAMPLERATE1024 = 1,
        SAMPLERATE512  = 2, 
        SAMPLERATE256  = 3,
        SAMPLERATE128  = 4,
        SAMPLERATE64   = 5
    } SampleRate;

    void ms_SetDebug(unsigned level);

    int ms_SendCommand(Command command, int arg, unsigned char *reply, int nReply);

    int ms_Open(const char *dev);
    int ms_Close();

    int ms_Ready();
    int ms_ReadStatus(SampleRate *sampleRate, BlockSize *blockSize, int *nDataBytes);

    int ms_SetBlockSize(BlockSize blockSize);
    int ms_ActualBlockSize(BlockSize blockSize);

    int ms_SetSampleRate(SampleRate sampleRate);
    int ms_ActualSampleRate(SampleRate sampleRate);

    int ms_ReadNextDataBlock(double *data, int nSamples);
    int ms_ReadAllDataBlocks(double *data, int nSamples);

    void ms_Dump(void *buffer, int count, char *str);
#endif
