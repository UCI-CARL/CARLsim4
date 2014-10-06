// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoCombinedContinue.h
// (c) Maarten Keijzer, GeNeura Team, 1999, 2000
/*
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Contact: todos@geneura.ugr.es, http://geneura.ugr.es
*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Modified by KDC @ UCI to allow for accessing generation number
//-----------------------------------------------------------------------------



#ifndef _pti_eoCombinedContinue_h
#define _pti_eoCombinedContinue_h

#include <eoCombinedContinue.h>

template< class EOT> 
class pti_eoCombinedContinue: public eoContinue<EOT> {
 public:

  /// Define Fitness
  typedef typename EOT::Fitness FitnessType;

  /// Ctor, make sure that at least on continuator is present
 pti_eoCombinedContinue( eoContinue<EOT>& _cont)
   : eoContinue<EOT> ()
    {
      continuators.push_back(&_cont);
    }

  /// Ctor - for historical reasons ... should disspear some day
 pti_eoCombinedContinue( eoContinue<EOT>& _cont1, eoContinue<EOT>& _cont2)
   : eoContinue<EOT> ()
    {
      continuators.push_back(&_cont1);
      continuators.push_back(&_cont2);
    }

  void add(eoContinue<EOT> & _cont)
  {
    continuators.push_back(&_cont);
  }


  ///////////// RAMON'S CODE ///////////////
  void removeLast(void)
  {
    continuators.pop_back();
  }
  ///////////// RAMON'S CODE (end) ///////////////
  
  // added this to return a pointer to the continuators object so the
  // generation number can be accessed. -- KDC
  std::vector<eoContinue<EOT>*>  returnContPtr()
  {
    return continuators;
  }

  /** Returns false when one of the embedded continuators say so (logical and)
   */
  virtual bool operator() ( const eoPop<EOT>& _pop )
  {

    for (unsigned i = 0; i < continuators.size(); ++i)
      if ( !(*continuators[i])(_pop) ) return false;
    return true;
  }

  virtual std::string className(void) const { return "pti_eoCombinedContinue"; }

 private:
  std::vector<eoContinue<EOT>*>    continuators;
};

#endif
