//-----------------------------------------------------------------------------
// Modified by KDC @ UCI to allow for accessing generation number
//-----------------------------------------------------------------------------
/*
  This class inherits from the eoGenContinue class and adds a single
  accessor for the private variable thisGeneration.
*/
#ifndef _pti_eoGenContinue_h
#define _pti_eoGenContinue_h

#include <eoGenContinue.h>

template< class EOT>
class pti_eoGenContinue: public eoGenContinue<EOT>
{
 public:
 
 pti_eoGenContinue( unsigned long _totalGens)
   : eoValueParam<unsigned>(0, "Generations", "Generations"),
    repTotalGenerations( _totalGens ),
    thisGenerationPlaceHolder(0),
    thisGeneration(thisGenerationPlaceHolder)
    {};

 pti_eoGenContinue( unsigned long _totalGens, unsigned long& _currentGen)
   : eoValueParam<unsigned>(0, "Generations", "Generations"),
    repTotalGenerations( _totalGens ),
    thisGenerationPlaceHolder(0),
    thisGeneration(_currentGen)
    {};

  virtual bool operator() ( const eoPop<EOT>& _vEO ) {
    (void)_vEO;
    thisGeneration++;
    value() = thisGeneration;
 
    if (thisGeneration >= repTotalGenerations)
      {
	eo::log << eo::logging << "STOP in eoGenContinue: Reached maximum number of generations [" << thisGeneration << "/" << repTotalGenerations << "]\n";
	return false;
      }
    return true;
  }

  virtual void totalGenerations( unsigned long _tg ) {
    repTotalGenerations = _tg;
    thisGeneration = 0;
  };
 
  virtual unsigned long totalGenerations( )
  {
    return repTotalGenerations;
  };
 
 
  virtual std::string className(void) const { return "eoGenContinue"; }
 
  void readFrom (std :: istream & __is) {
 
    __is >> thisGeneration; /* Loading the number of generations counted */
  }
 
  void printOn (std :: ostream & __os) const {
 
    __os << thisGeneration << std :: endl; /* Saving the number of generations counted */
  }
 
 private:
  unsigned long repTotalGenerations;
  unsigned long thisGenerationPlaceHolder;
  unsigned long& thisGeneration;
};
 
#endif
