#ifndef activetwo_h 
    #define activetwo_h

    typedef enum {
        BS_SPEED_MODE_4 = 4,
        BS_SPEED_MODE_5 = 5,
        BS_SPEED_MODE_6 = 6,
        BS_SPEED_MODE_7 = 7
    } bs_speedMode_t;

    extern bs_speedMode_t bs_speedMode;
    extern unsigned bs_nChan;
    extern unsigned bs_sampRate;
    extern unsigned bs_scansPerPoll;


    int bs_setSpeedMode(bs_speedMode_t speedMode);
    int bs_setScansPerPoll(unsigned scans);

    int bs_open();
    int bs_close();

    int bs_start();
    int bs_stop();

    int bs_poll(double *pollBuffer);
    //int bs_poll(char *pollBuffer);
#endif
