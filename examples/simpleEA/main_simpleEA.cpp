// (c) 2012 Regents of the University of California. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// UCI Documentation:
// -----------------------------------------------------------------------------
/* Date: 11/05/2012
   Kris Carlson:
   Toy Program to that evolves a genotypic string of alternating 1's and 0's.  
   The program uses a library called Evolving Objects and a modularity library
   that allows the user to run components of the library in parallel if needed.
*/
// -----------------------------------------------------------------------------

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#include <math.h>
// This is the only header file that needs to be included
// to use our EO modularity library.
#include <pti.h>

#define GENOME_SIZE 5

using namespace std;

// Our fitness function where all the testing and fitness assignments occur.
// This fitness function just creates a genome that has alternating one's 
// and zero's as its values.
int evaluateFitness(float* _genome, int _genomeSize, double& _fitness)
{
  double tempSum=0;
  double sum = 0;
  
  for(int i=0; i < _genomeSize; i++){
    if(i%2 == 0)
      {
	tempSum=fabs(_genome[i]);
      }
    else
      {
	tempSum=fabs((_genome[i]-1));
      }
    sum=sum+tempSum;
  }
  
  _fitness = sum;
  
  return 0;
}

// Our main function that handles the EO mechanics.
int main_fitness_function(int argc, char *argv[])
{
  float theGenome[GENOME_SIZE];

  // our fitness value!
  double fitness;
  
  // This is the name of the paramter file we are using. EO has many predefined 
  // files/algorithms so that we can choose different algorithms just by 
  // choosing different files.  This must always be run first.
  InitializeParamTuning("examples/simpleEA/ESEA-comma.param");
  // Must create a paramTuning object so that we can run the EA.
  
  // 1) The first argument is the file where evolving objects outputs the 
  // fitness and parameters.  This can and should be left blank for large
  // parameter sets.  The argument of "" can also be passed to signify an
  // empty string.
  
  // 2) The second argument is whether or not to output the best individual
  // for each generation in the resDir variable in the parameter file.
  // It should be used for larger parameter sets. It is off by default.
  system("mkdir -p results/simpleEA");
  ParamTuning *p = new ParamTuning("results/simpleEA/eoOutput.txt", true);
  //ParamTuning *p = new ParamTuning(false);

  string s1;
  // example of adding parameters
  char tmpFileName1[100];
  for(int i=0;i<GENOME_SIZE;i++){
    sprintf(tmpFileName1,"v%i",i);
    s1=tmpFileName1;
    p->addParam(s1,0,10.0);
  }

  // must be run *after* you load all the parameters.
  p->updateParameters();
  
  // let ourselves know we added the parameters!
  printf("Parameters added\n");

  // to keep track of the generations.
  int genCounter=0;
  // get number of individuals in population so we know how many individuals we
  // need to loop over to go over an entire generation.
  int populationSize;
  populationSize=p->getPopulationSize();
  
  printf("genCounter=%d\n",genCounter);
  
  cout << "p->maxGen=" << p->maxGen << endl;
  
  // loop over generations
  while (genCounter < p->maxGen){
    printf("Parameter initialization for generation %d\n",genCounter);
    genCounter++;    
  
    uint32_t IndiId=0;
    // Loop over the individuals in the population  
    while(IndiId < p->getPopulationSize() ){
      // example of getting the parameter values generated by EO
      for(int i=0;i<GENOME_SIZE;i++){
	sprintf(tmpFileName1,"v%i",i);
	s1=tmpFileName1;
	theGenome[i]=p->getParam(IndiId,s1);
      }
      
      // evaluate the fitness for this particular invidividual 
      evaluateFitness(theGenome,GENOME_SIZE,fitness);
      
      // assign the fitness (passes the fitness and IndiId to EO to deal with)
      p->setFitness(fitness, IndiId);
     
      IndiId++;
    } // end loop over individuals
  
    // run the evolutionary algorithm as each individual in the population
    // has now been asigned a fitness.  runEA() will return a boolean saying
    // whether or not to continue the evolutionary algorithm to the next generation.
    bool continueIter = p->runEA(); 
    // Stopping condition reached...
    if(!continueIter){
      fprintf(stderr," Stop condition reached...\n");
      break;
    }
  } // end while loop over generations

  // output our final population and generation number to the terminal
  p->printSortedPopulation();
  // genCounter-1 because we never ran the last generation.
  printf("genCounter=%d\n", genCounter-1);
  
  // delete the dynamically created variable
  delete p;

  return 0;
}

// A main that catches the exceptions for better organization
int main(int argc, char **argv)
{
  try
    {
      main_fitness_function(argc, argv);
    }
  catch (exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }
  
  return 1;
}

