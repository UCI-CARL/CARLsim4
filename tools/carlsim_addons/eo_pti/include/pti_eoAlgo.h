// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoAlgo.h
// (c) GeNeura Team, 1998
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

//UCI Documentation:
//-----------------------------------------------------------------------------
/* Kris Carlson 10/31/2011
   This class inherits from the original eoUF class and is analogous to the 
   eoAlgo.h file.  It adds the appropriate functions prototypes to implement 
   eoSlaveEasyEA.h.  These functions were taken from the () operator in the
   eoEasyEA.h header file.
*/
//-----------------------------------------------------------------------------

#ifndef _EOMODLIBALGO_H
#define _EOMODLIBALGO_H

#include <eoPop.h>                   // for population
#include <eoFunctor.h>

/**
    This is a generic class for population-transforming algorithms. There
    is only one operator defined, which takes a population and does stuff to
    it. It needn't be a complete algorithm, can be also a step of an
    algorithm. This class just gives a common interface to linear
    population-transforming algorithms.
*/

template< class EOT >
class eoModlibAlgo : public eoUF<eoPop<EOT>&, void>
{
public:
  virtual void getOffspring (eoPop<EOT>& _pop, eoPop<EOT>& _offspring) = 0;//{};
  virtual bool evaluatePopulation (eoPop<EOT>& _pop) = 0;//{return true;};
  virtual bool evaluateAndReplacePopulation (eoPop<EOT>& _pop, eoPop<EOT>& _offspring) = 0;//{return true;};
};

#endif
