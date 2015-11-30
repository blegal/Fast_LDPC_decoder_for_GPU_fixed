#ifndef CLASS_CTerminal
#define CLASS_CTerminal

#include "../timer/CTimer.h"
#include "../ber_analyzer/CErrorAnalyzer.h"


using namespace std;

class CTerminal
{
private:
    void ShowTime(unsigned long secondes);
    
protected:
	int fer_limit;
    double Eb_N0;
    CErrorAnalyzer *counter;
    CTimer         *timer;

public:
    CTerminal(CErrorAnalyzer *_counter, CTimer *_timer, double _eb_n0);

    void temp_report();
    void final_report();
};

#endif
