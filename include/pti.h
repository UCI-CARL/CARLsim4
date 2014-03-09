//UCI Documentation:
//-----------------------------------------------------------------------------
/* Kris Carlson 01/28/2013
   This file defines the ParamTuning class.  It holds the appropriate vector
   types that get passed to EO algorithms.  
 */
//-----------------------------------------------------------------------------

#ifndef PARAMTUNING_H
#define PARAMTUNING_H

#include <algorithm>
#include <string>
#include <iostream>
#include <iterator>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <cassert>
#include <cstring>

//need to include <sstream> to output generation data
#include <sstream>
//to do system commands
#include <cstdlib>

using namespace std;

#include <eo>

// representation specific
#include "pti_eoAlgo.h"
#include "pti_eoEasyEA.h"
#include "pti_eoCombinedContinue.h"
#include "pti_make_es.h"
#include "pti_make_real.h"

class EO_Params
{
 public:
  EO_Params(double _min, double _max) {
    min  = _min;
    max  = _max;
  }
  double min, max;
  int    id;
};

typedef std::map<string, EO_Params*> t_paramMap;

typedef eoMinimizingFitness  FitT;

typedef eoVector<FitT, double> fitVector;

bool runReal=false;
bool runFull=false;
bool runStdev=false;
bool runSimple=false;
string  configFile;
double* fitnessValue;
double* oldFitnessValue;
bool*   fitnessExists;
//eoReal Genotypes
eoModlibAlgo< eoReal<FitT> >* real_my_ga;
eoPop <eoReal<FitT> >* real_my_pop;
eoPop <eoReal<FitT> >  real_my_offspring;
pti_eoCombinedContinue<eoReal<FitT> >* real_my_continuator;
//eoEsSimple Genotypes
eoModlibAlgo< eoEsSimple<FitT> >* simple_my_ga;
eoPop <eoEsSimple<FitT> >* simple_my_pop;
eoPop <eoEsSimple<FitT> >  simple_my_offspring;
pti_eoCombinedContinue<eoEsSimple<FitT> >* simple_my_continuator;
//eoEsStdev Genotypes
eoModlibAlgo< eoEsStdev<FitT> >* stdev_my_ga;
eoPop <eoEsStdev<FitT> >* stdev_my_pop;
eoPop <eoEsStdev<FitT> >  stdev_my_offspring;
pti_eoCombinedContinue<eoEsStdev<FitT> >* stdev_my_continuator;
//eoEsFull Genotypes
eoModlibAlgo< eoEsFull<FitT> >* full_my_ga;
eoPop <eoEsFull<FitT> >* full_my_pop;
eoPop <eoEsFull<FitT> >  full_my_offspring;
pti_eoCombinedContinue<eoEsFull<FitT> >* full_my_continuator;

int popSize;
int globalMaxGen;
FILE* fpEOCopy;
string outputFile;//name of file to which we output data
string paramFile; //initialized by InitializeParamTuning
bool firstEvaluation;
int  numPopulationWithFitness; // This counts the number of population that already has a fitness function.
bool outputToFile; //if set to true will output info to desired file location
unsigned int vecSize;
t_paramMap paramList; 

class ParamTuning {
   
 public:
//change the order here so older files will work
  ParamTuning(string _outputFile="", bool _outputBestIndi=false);
  
  void updateParameters();
  bool runEA();
  void addParam(string strName, double _min, double _max);
  //check to see if an individual already has a fitness assigned to it
  bool fitnessAlreadyExists(uint32_t IndiId);
  //return the value of the parameter generated in EO
  double getParam(uint32_t IndiId, string _paramName);
  double getParamReal(uint32_t _IndiId, string _paramName);
  double getParamSimple(uint32_t _IndiId, string _paramName);
  double getParamStdev(uint32_t _IndiId, string _paramName);
  double getParamFull(uint32_t _IndiId, string _paramName);
  int getVectorSize();
  int getMaxGen();
  //pass the individual fitness value to EO framework
  //changed so that it is a pointer to a fitness value so that an array
  //of fitnesses can be passed to it.
  void setFitness(double* _fitnessValue, uint32_t _IndiId, int _numConfig); //TODO: fix this
  // pass the individual fitness value to EO framework.
  void setFitness(double _fitness, uint32_t _IndiId);
  uint32_t getPopulationSize();

  //prints the parent population sorted by fitness
  void printSortedPopulation();
  
  eoParser* parser;
  eoState*  state;

  //variables used in updateParameters()
  bool       parameterUpdated;//initialized in constructor
  int        maxGen;

  bool       outputBestIndi; 
};

void InitializeParamTuning(string _configFile)                            
{
  
  char* argParam[2];
  string argParamFile;
  string argParam0;
  paramFile=_configFile;
  cout << "paramFile is " << paramFile << endl;
  argParam0 = "UCI-EO";
  argParam[0] = strdup(argParam0.c_str());
  argParamFile = "@" + _configFile;
  argParam[1] = strdup(argParamFile.c_str());
  outputToFile=false;
  
  eoParser tmp_parser(2, argParam); // for user-parameter reading                  
  eoState tmp_state; // keeps all things allocated                                  
  
  //grab the population size from the parameter file for initialization purposes
  popSize = tmp_parser.getORcreateParam(unsigned(20), "popSize", "Population Size", 'P', "Evolution Engine").value();
  
  //grab the maximum generation the simulation reaches from the paramter file
  globalMaxGen=tmp_parser.getORcreateParam(unsigned(100), "maxGen", "Maximum number of generations () = none)",'G',"Stopping criterion").value();

  cout << "popsize=" << popSize << " " << "globalMaxGen=" << globalMaxGen << endl;

  // Run the appropriate algorithm (From four genotype options)                 
  if(tmp_parser.getORcreateParam(true, "Isotropic", "Isotropic self-adaptive mutation", 'i', "ES mutation").value() == false)                                     
    {                                                                           
      cout << "Using eoReal" << endl;                                           
      runReal=true;                                                             
      return;                                                                   
    }                                                                           
  else if(tmp_parser.getORcreateParam(false, "Stdev", "One self-adaptive stDev per variable", 's', "ES mutation").value() == false)                               
    {                                                                           
      cout << "Using eoEsSimple" << endl;                                         
      runSimple=true;                                                           
      return;                                                                   
    }                                                                           
  else if(tmp_parser.getORcreateParam(false, "Correl", "Use correlated mutations", 'c', "ES mutation").value() == false)                                          
    {                                                                           
      cout << "Using eoEsStdev" << endl;                                          
      runStdev=true;                                                            
      return;                                                                   
    }                                                                           
  else                                                                          
    {                                                                           
      cout << "Using eoEsFull" << endl;                                           
      runFull=true;                                                             
      return;                                                                   
    }                                                                           
}


void print_vector (FILE* fp, const std::vector<double>& _ind, const char* _name)
{
  fprintf(fp, "%s : ", _name);
  for (unsigned i = 0; i < _ind.size(); i++)
    fprintf(fp, " %f ", _ind[i]);
}

// writes parameters (in alpha order) to hard drive in binary format
void write_vector (const std::vector<double>& _ind)
{
  FILE* pFile;
  pFile = fopen ( "indi_vector.bin" , "ab" );
  
  for (unsigned i = 0; i < _ind.size(); i++){
    double tmp = _ind[i];
    fwrite(&tmp,sizeof(tmp),1,pFile);
  }
  fclose(pFile);
}
// writes parameters (in alpha order) to hard drive in binary format takes generation 
// number and output directory  as arguments for this prototype.
void write_vector (const std::vector<double>& _ind, string _genNum, string _dirName)
{
  FILE* pFile;
  string fileName =  _dirName + "/best_ind_gen_" + _genNum + ".dat";
  pFile = fopen ( fileName.c_str(), "wb" );
  for (unsigned i = 0; i < _ind.size(); i++){
    double tmp = _ind[i];
    fwrite(&tmp,sizeof(tmp),1,pFile);
  }
  fclose(pFile);
}

string getGenReal()
{
  // have to iterate through the vector to get the correct continuator type,
  // which is eoGenContinue.
  std::vector<eoContinue<eoReal<FitT> >*> theIterator = real_my_continuator->returnContPtr();
  string genString;
  // iterate until we find eoGenContinue
  for(unsigned int i=0;i<theIterator.size();i++){
    // when true, we extract the generation
    if(dynamic_cast<eoGenContinue<eoReal<FitT> >*> (theIterator.at(i)) != NULL){
      stringstream oss;
      (theIterator.at(i))->printOn(oss);
      genString=oss.str();
      // remove the newline at the end appended by the eo printOn function
      genString=genString.substr(0, genString.size()-1);
      break;
    }
  }
  if(genString.empty()){
    fprintf(stderr,"Did not find eoGenContinue object");
    assert(!genString.empty());
  }
  return genString;
}

string getGenSimple()
{
  // have to iterate through the vector to get the correct continuator type,
  // which is eoGenContinue.
  std::vector<eoContinue<eoEsSimple<FitT> >*> theIterator = simple_my_continuator->returnContPtr();
  string genString;
  // iterate until we find eoGenContinue
  for(unsigned int i=0;i<theIterator.size();i++){
    // when true, we extract the generation
    if(dynamic_cast<eoGenContinue<eoEsSimple<FitT> >*> (theIterator.at(i)) != NULL){
      stringstream oss;
      (theIterator.at(i))->printOn(oss);
      genString=oss.str();
      // remove the newline at the end appended by the eo printOn function
      genString=genString.substr(0, genString.size()-1);
      break;
    }
  }
  if(genString.empty()){
    fprintf(stderr,"Did not find eoGenContinue object");
    assert(!genString.empty());
  }
  return genString;
}

string getGenStdev()
{
  // have to iterate through the vector to get the correct continuator type,
  // which is eoGenContinue.
  std::vector<eoContinue<eoEsStdev<FitT> >*> theIterator = stdev_my_continuator->returnContPtr();
  string genString;
  // iterate until we find eoGenContinue
  for(unsigned int i=0;i<theIterator.size();i++){
    // when true, we extract the generation
    if(dynamic_cast<eoGenContinue<eoEsStdev<FitT> >*> (theIterator.at(i)) != NULL){
      stringstream oss;
      (theIterator.at(i))->printOn(oss);
      genString=oss.str();
      // remove the newline at the end appended by the eo printOn function
      genString=genString.substr(0, genString.size()-1);
      break;
    }
  }
  if(genString.empty()){
    fprintf(stderr,"Did not find eoGenContinue object");
    assert(!genString.empty());
  }
  return genString;
}

string getGenFull()
{
  // have to iterate through the vector to get the correct continuator type,
  // which is eoGenContinue.
  std::vector<eoContinue<eoEsFull<FitT> >*> theIterator = full_my_continuator->returnContPtr();
  string genString;
  // iterate until we find eoGenContinue
  for(unsigned int i=0;i<theIterator.size();i++){
    // when true, we extract the generation
    if(dynamic_cast<eoGenContinue<eoEsFull<FitT> >*> (theIterator.at(i)) != NULL){
      stringstream oss;
      (theIterator.at(i))->printOn(oss);
      genString=oss.str();
      // remove the newline at the end appended by the eo printOn function
      genString=genString.substr(0, genString.size()-1);
      break;
    }
  }
  if(genString.empty()){
    fprintf(stderr,"Did not find eoGenContinue object");
    assert(!genString.empty());
  }
  return genString;
}

// function to output the best individual
void outputBest(bool outputBestIndi, ParamTuning* p)
{
  if(outputBestIndi){
    // grab directory name from parser info:
    eoParam* dirNameParam = p->parser->getParamWithLongName("resDir");
    // cast to correct inherited class of eoParam, which is
    dirNameParam = static_cast<eoValueParam<std::string>* >(dirNameParam);
    string dirName = dirNameParam->getValue();
    char cmdName[100];
    sprintf(cmdName, "mkdir -p %s", dirName.c_str());
    int response = system(cmdName);
    assert(response == 0);
   
    if(runReal == true){
      string genString = getGenReal();
      // grab the best-performing individual
      const eoReal<FitT>  bestIndi = real_my_pop->best_element();
      write_vector(bestIndi,genString,dirName);
    }
    else if(runSimple == true){
      string genString = getGenSimple();
      // grab the best-performing individual
      const eoEsSimple<FitT>  bestIndi = simple_my_pop->best_element();
      write_vector(bestIndi,genString,dirName);
    }
    else if(runStdev == true){
      string genString = getGenStdev();
      // grab the best-performing individual
      const eoEsStdev<FitT>  bestIndi = stdev_my_pop->best_element();
      write_vector(bestIndi,genString,dirName);
    }
    else{
      string genString = getGenFull();
      // grab the best-performing individual
      const eoEsFull<FitT>  bestIndi = full_my_pop->best_element();
      write_vector(bestIndi,genString,dirName);
    }    
  }
  return;
}
  
//4 different getIndividualId functions
//For eoReal:
int getIndividualIdReal(const std::vector<double>& _ind, bool useParent)
{
  uint32_t pSize, vSize;
  if(useParent) {
    assert (real_my_pop != NULL);//This makes sure the pop exists! -- Kris
    pSize = (*real_my_pop).size();
    vSize   = (*real_my_pop)[0].size();
  }
  else {
    pSize = real_my_offspring.size();
    vSize   = real_my_offspring[0].size();
  }

  // number of variables/index in _ind and given population should always be same.
  assert(vSize == _ind.size());

  for (uint32_t popId=0; popId < pSize; popId++) {
    bool matched = false;
    for (unsigned i = 0; i < _ind.size(); i++) {
      double diffVal;
      if (useParent)
	diffVal = ((*real_my_pop)[popId][i]-_ind[i]);
      else
	diffVal = (real_my_offspring[popId][i]-_ind[i]);

      if (abs(diffVal) > 0.00001) {
	matched = false;
	break;
      }
      else
	matched = true;
    }
    if (matched)
      return popId;
  }

  return -1;
}

//For eoEsSimple
int getIndividualIdSimple(const std::vector<double>& _ind, bool useParent)
{
  uint32_t pSize, vSize;
  if(useParent) {
    assert (simple_my_pop != NULL);//This makes sure the pop exists! -- Kris
    pSize = (*simple_my_pop).size();
    vSize   = (*simple_my_pop)[0].size();
  }
  else {
    pSize = simple_my_offspring.size();
    vSize   = simple_my_offspring[0].size();
  }

  // number of variables/index in _ind and given population should always be same.
  assert(vSize == _ind.size());

  for (uint32_t popId=0; popId < pSize; popId++) {
    bool matched = false;
    for (unsigned i = 0; i < _ind.size(); i++) {
      double diffVal;
      if (useParent)
	diffVal = ((*simple_my_pop)[popId][i]-_ind[i]);
      else
	diffVal = (simple_my_offspring[popId][i]-_ind[i]);

      if (abs(diffVal) > 0.00001) {
	matched = false;
	break;
      }
      else
	matched = true;
    }
    if (matched)
      return popId;
  }

  return -1;
}

//For eoEsStdev
int getIndividualIdStdev(const std::vector<double>& _ind, bool useParent)
{
  uint32_t pSize, vSize;
  if(useParent) {
    assert (stdev_my_pop != NULL);//This makes sure the pop exists! -- Kris
    pSize = (*stdev_my_pop).size();
    vSize   = (*stdev_my_pop)[0].size();
  }
  else {
    pSize = stdev_my_offspring.size();
    vSize   = stdev_my_offspring[0].size();
  }

  // number of variables/index in _ind and given population should always be same.
  assert(vSize == _ind.size());

  for (uint32_t popId=0; popId < pSize; popId++) {
    bool matched = false;
    for (unsigned i = 0; i < _ind.size(); i++) {
      double diffVal;
      if (useParent)
	diffVal = ((*stdev_my_pop)[popId][i]-_ind[i]);
      else
	diffVal = (stdev_my_offspring[popId][i]-_ind[i]);

      if (abs(diffVal) > 0.00001) {
	matched = false;
	break;
      }
      else
	matched = true;
    }
    if (matched)
      return popId;
  }

  return -1;
}

//For eoEsFull
int getIndividualIdFull(const std::vector<double>& _ind, bool useParent)
{
  uint32_t pSize, vSize;
  if(useParent) {
    assert (full_my_pop != NULL);//This makes sure the pop exists! -- Kris
    pSize = (*full_my_pop).size();
    vSize   = (*full_my_pop)[0].size();
  }
  else {
    pSize = full_my_offspring.size();
    vSize   = full_my_offspring[0].size();
  }

  // number of variables/index in _ind and given population should always be same.
  assert(vSize == _ind.size());

  for (uint32_t popId=0; popId < pSize; popId++) {
    bool matched = false;
    for (unsigned i = 0; i < _ind.size(); i++) {
      double diffVal;
      if (useParent)
	diffVal = ((*full_my_pop)[popId][i]-_ind[i]);
      else
	diffVal = (full_my_offspring[popId][i]-_ind[i]);

      if (abs(diffVal) > 0.00001) {
	matched = false;
	break;
      }
      else
	matched = true;
    }
    if (matched)
      return popId;
  }

  return -1;
}

int getIndividualId(const std::vector<double>& _ind, bool useParent)
{
  if(runReal == true)
    {
      return getIndividualIdReal(_ind, useParent);
    }
  else if(runSimple == true)
    {
      return getIndividualIdSimple(_ind, useParent);
    }
  else if(runStdev == true)
    {
      return getIndividualIdStdev(_ind, useParent);
    }
  else
    {
      return getIndividualIdFull(_ind, useParent);
    }
}

//This makes it so you don't have to run the individual through the fitness condition
//again.  You want to avoid doing this because evaluating the fitness condition of an
//individual is the most expensive part of the process.
void populationWithFitnessReal()
{
  numPopulationWithFitness = 0;
  memcpy (oldFitnessValue, fitnessValue, sizeof(double)*popSize*2);
  if(outputToFile)
    {
      fpEOCopy=fopen(outputFile.c_str(),"a");
      fprintf(fpEOCopy, "Population With Fitness Evaluation...\n");
      fprintf(fpEOCopy, "Size of new offspring population... %zu\n", real_my_offspring.size());
    }
  for(uint32_t i=0; i < real_my_offspring.size(); i++) {
    if(outputToFile)
      {
	char popStr[100];
	sprintf(popStr,"offspring : %d", i);
	print_vector (fpEOCopy, real_my_offspring[i], popStr);
      }
    int popId = getIndividualIdReal(real_my_offspring[i], true);
    if (popId == -1) {
      fitnessExists[i] = false;// this says that we do have to evaluate the population
      
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [NEW]  \tNew offspring population %d\n", i);
	}
    }
    else {
      numPopulationWithFitness++;
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [OLD] \toffspring population %d already exists as %d \n", i, popId);
	}
      fitnessValue[i]=oldFitnessValue[popId];
      fitnessExists[i] = true; //this says that we don't have to evaluate the population again
    }
  }

  if(outputToFile)
    fclose(fpEOCopy);

  return;
}


void populationWithFitnessSimple()
{
  numPopulationWithFitness = 0;
  memcpy (oldFitnessValue, fitnessValue, sizeof(double)*popSize*2);
  if(outputToFile)
    {
      fpEOCopy=fopen(outputFile.c_str(),"a");
      fprintf(fpEOCopy, "Population With Fitness Evaluation...\n");
      fprintf(fpEOCopy, "Size of new offspring population... %zu\n", simple_my_offspring.size());
    }
  for(uint32_t i=0; i < simple_my_offspring.size(); i++) {
    if(outputToFile)
      {
	char popStr[100];
	sprintf(popStr,"offspring : %d", i);
	print_vector (fpEOCopy, simple_my_offspring[i], popStr);
      }
    int popId = getIndividualIdSimple(simple_my_offspring[i], true);
    if (popId == -1) {
      fitnessExists[i] = false;// this says that we do have to evaluate the population
      
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [NEW]  \tNew offspring population %d\n", i);
	}
    }
    else {
      numPopulationWithFitness++;
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [OLD] \toffspring population %d already exists as %d \n", i, popId);
	}
      fitnessValue[i]=oldFitnessValue[popId];
      fitnessExists[i] = true; //this says that we don't have to evaluate the population again
    }
  }

  if(outputToFile)
    fclose(fpEOCopy);

  return;
}


void populationWithFitnessStdev()
{
  numPopulationWithFitness = 0;
  memcpy (oldFitnessValue, fitnessValue, sizeof(double)*popSize*2);
  if(outputToFile)
    {
      fpEOCopy=fopen(outputFile.c_str(),"a");
      fprintf(fpEOCopy, "Population With Fitness Evaluation...\n");
      fprintf(fpEOCopy, "Size of new offspring population... %zu\n", stdev_my_offspring.size());
    }
  for(uint32_t i=0; i < stdev_my_offspring.size(); i++) {
    if(outputToFile)
      {
	char popStr[100];
	sprintf(popStr,"offspring : %d", i);
	print_vector (fpEOCopy, stdev_my_offspring[i], popStr);
      }
    int popId = getIndividualIdStdev(stdev_my_offspring[i], true);
    if (popId == -1) {
      fitnessExists[i] = false;// this says that we do have to evaluate the population
      
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [NEW]  \tNew offspring population %d\n", i);
	}
    }
    else {
      numPopulationWithFitness++;
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [OLD] \toffspring population %d already exists as %d \n", i, popId);
	}
      fitnessValue[i]=oldFitnessValue[popId];
      fitnessExists[i] = true; //this says that we don't have to evaluate the population again
    }
  }

  if(outputToFile)
    fclose(fpEOCopy);

  return;
}

void populationWithFitnessFull()
{
  numPopulationWithFitness = 0;
  memcpy (oldFitnessValue, fitnessValue, sizeof(double)*popSize*2);
  if(outputToFile)
    {
      fpEOCopy=fopen(outputFile.c_str(),"a");
      fprintf(fpEOCopy, "Population With Fitness Evaluation...\n");
      fprintf(fpEOCopy, "Size of new offspring population... %zu\n", full_my_offspring.size());
    }
  for(uint32_t i=0; i < full_my_offspring.size(); i++) {
    if(outputToFile)
      {
	char popStr[100];
	sprintf(popStr,"offspring : %d", i);
	print_vector (fpEOCopy, full_my_offspring[i], popStr);
      }
    int popId = getIndividualIdFull(full_my_offspring[i], true);
    if (popId == -1) {
      fitnessExists[i] = false;// this says that we do have to evaluate the population
      
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [NEW]  \tNew offspring population %d\n", i);
	}
    }
    else {
      numPopulationWithFitness++;
      if(outputToFile)
	{
	  fprintf(fpEOCopy, " [OLD] \toffspring population %d already exists as %d \n", i, popId);
	}
      fitnessValue[i]=oldFitnessValue[popId];
      fitnessExists[i] = true; //this says that we don't have to evaluate the population again
    }
  }

  if(outputToFile)
    fclose(fpEOCopy);

  return;
}

//return the real value of the fitness for the given list of vector...
//Need to print these values out again.
double real_value (const std::vector<double>& _ind)
{
  if(outputToFile){
    fpEOCopy=fopen(outputFile.c_str(),"a");
    fprintf(fpEOCopy, "real_value function called and population is being evaluated\n");
    fprintf(fpEOCopy, "real_value Population : using %s\t", firstEvaluation?"parent population":"offspring population");
  }
  int IndiId = getIndividualId(_ind, firstEvaluation);
  
  if (IndiId >= 0) 
    {
      if(outputToFile)
	{
	  string blankCString = "";
	  print_vector(fpEOCopy, _ind, blankCString.c_str());
	  fprintf(fpEOCopy, "Fitness: %f (matching population = %d)\n", fitnessValue[IndiId], IndiId);
	}
      if(outputToFile)
	fclose(fpEOCopy);
      
      //Jay originally had: //TODO: Confirm this is correct
      //return (1/fitnessValue[IndiId]); //to maximize I guess
      return (fitnessValue[IndiId]); //to minimize
    }
  assert(0);
  return 0.0;
}

//set up the E-O parameters for the evolutionary algorithm
template<class EOT> 
void setAlgorithmSimple(EOT, eoParser& _parser, eoState& _state, ParamTuning* p)
{
  typedef typename EOT::Fitness FitT;
  
  //////////////////////////////////////////////////////
  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

  // The evaluation fn - encapsulated into an eval counter for output
  eoEvalFuncPtr<EOT, double, const std::vector<double>&>*  mainEval = 
    new eoEvalFuncPtr<EOT, double, const std::vector<double>&> ( real_value );
  
  eoEvalFuncCounter<EOT>* eval = new eoEvalFuncCounter<EOT>(*mainEval);
  // the genotype - through a genotype initializer
  
  eoRealInitBounded<EOT>& init = make_genotype(_parser, _state, EOT());
  // Build the variation operator (any seq/prop construct)
  eoGenOp<EOT>& op = make_op(_parser, _state, init);
  
    
  //// Now the representation-independent things
  //////////////////////////////////////////////

  // initialize the population - and evaluate
  // yes, this is representation indepedent once you have an eoInit
  eoPop<EOT>& pop = make_pop(_parser, _state, init);

  // apply<EOT> (eval, pop);//why isn't this included?
  //I assume this sorts the population for the first time
  //so we don't call it.
  //he has the next line instead of the above line
  
  //I think the print statement calls a function
  //that accesses my_pop
  
  //I have to assign the correct pop type
  
  simple_my_pop = &pop;//have to add this one and think about it
 
  //he then prints
  //printPopulation("Initial Population Before Apply\n",true, fpEOCopy);
  //can't call this yet.

  // stopping criteria
  eoContinue<EOT> & term = make_continue(_parser, _state, *eval);
  simple_my_continuator = static_cast<pti_eoCombinedContinue<eoEsSimple<FitT> >* >(&term);
  
  // output 
  eoCheckPoint<EOT> & checkpoint = make_checkpoint(_parser, _state, *eval, term);

  // algorithm (need the operator!)
  //I think this is where I make a modlib algo
  eoModlibAlgo<EOT>& ga = pti_make_algo_scalar(_parser, _state, *eval, checkpoint, op);
  
  //this is probably so he can call this outside this function
  simple_my_ga = &ga;

  ///// End of construction of the algorithm
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(_parser);
  
}

template<class EOT> 
void setAlgorithmStdev(EOT, eoParser& _parser, eoState& _state, ParamTuning* p)
{
  typedef typename EOT::Fitness FitT;

  //////////////////////////////////////////////////////
  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

  // The evaluation fn - encapsulated into an eval counter for output
  eoEvalFuncPtr<EOT, double, const std::vector<double>&>*  mainEval = 
    new eoEvalFuncPtr<EOT, double, const std::vector<double>&> ( real_value );
  
   eoEvalFuncCounter<EOT>* eval = new eoEvalFuncCounter<EOT>(*mainEval);

  // the genotype - through a genotype initializer
  eoRealInitBounded<EOT>& init = make_genotype(_parser, _state, EOT());
    
  // Build the variation operator (any seq/prop construct)
  eoGenOp<EOT>& op = make_op(_parser, _state, init);
  
  //// Now the representation-independent things
  //////////////////////////////////////////////

  // initialize the population - and evaluate
  // yes, this is representation indepedent once you have an eoInit
  eoPop<EOT>& pop = make_pop(_parser, _state, init);

  // apply<EOT> (eval, pop);//why isn't this included?
  //I assume this sorts the population for the first time
  //so we don't call it.
  //he has the next line instead of the above line
  
  //I think the print statement calls a function
  //that accesses my_pop
  
  //I have to assign the correct pop type
  stdev_my_pop = &pop;//have to add this one and think about it
 
  //he then prints
  //printPopulation("Initial Population Before Apply\n",true,fpEOCopy);
  //can't call this yet.

  // stopping criteria
  //eoContinue<eoEsStdev<FitT> >* stdev_my_continuator;
  // need to change this to make_continue_modlib etc.
  eoContinue<EOT> & term = make_continue(_parser, _state, *eval);
  stdev_my_continuator = static_cast<pti_eoCombinedContinue<eoEsStdev<FitT> >* >(&term);
  //REMOVE THIS: TODO -- KDC
  //CDerived * b = static_cast<CDerived*>(a);
  //stdev_my_continuator = make_continue(_parser, _state, *eval);
  //eoContinue<EOT> & term = stdev_my_continuator;

  // output 
  eoCheckPoint<EOT> & checkpoint = make_checkpoint(_parser, _state, *eval, term);

  // algorithm (need the operator!)
  //I think this is where I make a modlib algo
  eoModlibAlgo<EOT>& ga = pti_make_algo_scalar(_parser, _state, *eval, checkpoint, op);
  
  //this is probably so he can call this outside this function
  stdev_my_ga = &ga;

  ///// End of construction of the algorithm
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(_parser);
}

template<class EOT> 
void setAlgorithmFull(EOT, eoParser& _parser, eoState& _state, ParamTuning* p)
{
  typedef typename EOT::Fitness FitT;

  //////////////////////////////////////////////////////
  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

  // The evaluation fn - encapsulated into an eval counter for output
  eoEvalFuncPtr<EOT, double, const std::vector<double>&>*  mainEval = 
    new eoEvalFuncPtr<EOT, double, const std::vector<double>&> ( real_value );
  
   eoEvalFuncCounter<EOT>* eval = new eoEvalFuncCounter<EOT>(*mainEval);

  // the genotype - through a genotype initializer
  eoRealInitBounded<EOT>& init = make_genotype(_parser, _state, EOT());
    
  // Build the variation operator (any seq/prop construct)
  eoGenOp<EOT>& op = make_op(_parser, _state, init);
  
  //// Now the representation-independent things
  //////////////////////////////////////////////

  // initialize the population - and evaluate
  // yes, this is representation indepedent once you have an eoInit
  eoPop<EOT>& pop = make_pop(_parser, _state, init);

  // apply<EOT> (eval, pop);//why isn't this included?
  //I assume this sorts the population for the first time
  //so we don't call it.
  //he has the next line instead of the above line
  
  //I think the print statement calls a function
  //that accesses my_pop
  
  //I have to assign the correct pop type
  
  full_my_pop = &pop;//have to add this one and think about it
 
  //he then prints
  //printPopulation("Initial Population Before Apply\n",true,fpEOCopy);
  //can't call this yet.

  // stopping criteria
  eoContinue<EOT> & term = make_continue(_parser, _state, *eval);
  full_my_continuator = static_cast<pti_eoCombinedContinue<eoEsFull<FitT> >* >(&term);
  // output 
  eoCheckPoint<EOT> & checkpoint = make_checkpoint(_parser, _state, *eval, term);

  // algorithm (need the operator!)
  //I think this is where I make a modlib algo
  eoModlibAlgo<EOT>& ga = pti_make_algo_scalar(_parser, _state, *eval, checkpoint, op);
  
  //this is probably so he can call this outside this function
  full_my_ga = &ga;

  ///// End of construction of the algorithm
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(_parser);
  
}

template<class EOT> 
void setAlgorithmReal(EOT, eoParser& _parser, eoState& _state, ParamTuning* p)
{
  typedef typename EOT::Fitness FitT;

  //////////////////////////////////////////////////////
  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

  // The evaluation fn - encapsulated into an eval counter for output
  eoEvalFuncPtr<EOT, double, const std::vector<double>&>*  mainEval = 
    new eoEvalFuncPtr<EOT, double, const std::vector<double>&> ( real_value );
  
   eoEvalFuncCounter<EOT>* eval = new eoEvalFuncCounter<EOT>(*mainEval);

  // the genotype - through a genotype initializer
  eoRealInitBounded<EOT>& init = make_genotype(_parser, _state, EOT());
    
  // Build the variation operator (any seq/prop construct)
  eoGenOp<EOT>& op = make_op(_parser, _state, init);
  
  //// Now the representation-independent things
  //////////////////////////////////////////////

  // initialize the population - and evaluate
  // yes, this is representation indepedent once you have an eoInit
  eoPop<EOT>& pop = make_pop(_parser, _state, init);

  // apply<EOT> (eval, pop);//why isn't this included?
  //I assume this sorts the population for the first time
  //so we don't call it.
  //he has the next line instead of the above line
  
  //I think the print statement calls a function
  //that accesses my_pop
  
  //I have to assign the correct pop type
  
  real_my_pop = &pop;//have to add this one and think about it
 
  //he then prints
  //printPopulation("Initial Population Before Apply\n",true,fpEOCopy);
  //can't call this yet.

  // New test code  -- REMOVE
  // --------------------------------------------
  // need to make a variable with global scope to access output of
  // make_continue
  
  // stopping criteria
  eoContinue<EOT> & term = make_continue(_parser, _state, *eval);
  real_my_continuator = static_cast<pti_eoCombinedContinue<eoReal<FitT> >* >(&term);
    
  // End new test code
  // --------------------------------------------

  // output 
  eoCheckPoint<EOT> & checkpoint = make_checkpoint(_parser, _state, *eval, term);

  // algorithm (need the operator!)
  //I think this is where I make a modlib algo
  eoModlibAlgo<EOT>& ga = pti_make_algo_scalar(_parser, _state, *eval, checkpoint, op);
  
  //this is probably so he can call this outside this function
  real_my_ga = &ga;

  ///// End of construction of the algorithm
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(_parser);
  
}

// prints generation to output file where other info is
// stored, but only if output to file is not an empty string
void printGeneration()
{
  string genString;
  if(outputToFile){
    // first grab the correct generation number:
    if(runReal == true){
      genString = getGenReal();
    }
    else if(runSimple == true){
      genString = getGenSimple();
    }
    else if(runStdev == true){
      genString = getGenStdev();
    }
    else{
      genString = getGenFull();
    }
    fpEOCopy=fopen(outputFile.c_str(),"a");
    fprintf(fpEOCopy, "Running generation: %s\n", genString.c_str());
    fclose(fpEOCopy);
  }
}

ParamTuning::ParamTuning(string _outputFile, bool _outputBestIndi)
{
  outputBestIndi = _outputBestIndi;
  real_my_pop  = NULL;
  real_my_ga   = NULL;
  simple_my_pop  = NULL;
  simple_my_ga   = NULL;
  stdev_my_pop  = NULL;
  stdev_my_ga   = NULL;
  full_my_pop  = NULL;
  full_my_ga   = NULL;
  parser = NULL;
  state = NULL;
  vecSize = 0; // number of parameters per individiual
  // indicates if update parameters function has been called
  parameterUpdated=false;

  if(!_outputFile.empty())
    {
      outputFile=_outputFile;
      outputToFile=true;
    }

  if (popSize < 2) 
    {
      fprintf(stderr, "Population should be minimum of 2; Given popSize = %d\n", popSize);
      popSize = 2;
    }
  fitnessValue    = new double[popSize*2];
  oldFitnessValue = new double[popSize*2];
  fitnessExists = new bool[popSize*2];
  //globalMaxGen read from the param file using InitializeParamTuning
  maxGen=globalMaxGen;
  firstEvaluation=true;
}

// run the GA learning network
template<class EOT> 
bool runAlgorithmReal(EOT, ParamTuning* p)
{
  printGeneration();  
  bool continueES = true;

  if (firstEvaluation) {
    // evaluate the parent population...
    (real_my_ga)->evaluatePopulation(*(real_my_pop));
    cout << "Initial Population\n";
    real_my_pop->sortedPrintOn(cout);
    cout << endl;
    outputBest(p->outputBestIndi,p);
  }
  else {
    continueES = (real_my_ga)->evaluateAndReplacePopulation(*(real_my_pop), real_my_offspring);
    outputBest(p->outputBestIndi,p);
  }
  
  if (continueES) {
    // generate a new offspring..
    real_my_ga->getOffspring(*(real_my_pop), real_my_offspring);
    //compareParentOffspring(); //not used now
    populationWithFitnessReal();
  }
  else {
    printf("Final Population\n");
    real_my_pop->sortedPrintOn(cout);
    cout << endl;
  }

  firstEvaluation = false;
  return continueES;
}

//run the evolutionary algorithm
template<class EOT> 
bool runAlgorithmSimple(EOT, ParamTuning* p)
{
  printGeneration();
  bool continueES = true;

  if (firstEvaluation) {
    // evaluate the parent population...
    (simple_my_ga)->evaluatePopulation(*(simple_my_pop));
    cout << "Initial Population\n";
    simple_my_pop->sortedPrintOn(cout);
    cout << endl;
    outputBest(p->outputBestIndi,p);
  }
  else {
    continueES = (simple_my_ga)->evaluateAndReplacePopulation(*(simple_my_pop), simple_my_offspring);
    outputBest(p->outputBestIndi,p);
  }
  
  if (continueES) {
    // generate a new offspring..
    simple_my_ga->getOffspring(*(simple_my_pop), simple_my_offspring);
    //compareParentOffspring(); //not used now
    populationWithFitnessSimple();
  }
  else {
    printf("Final Population\n");
    simple_my_pop->sortedPrintOn(cout);
    cout << endl;
  }

  firstEvaluation = false;
  return continueES;
}

// run the evolutionary algorithm
template<class EOT> 
bool runAlgorithmStdev(EOT, ParamTuning* p)
{
  printGeneration();
  bool continueES = true;
  if (firstEvaluation) {
    // evaluate the parent population...
    (stdev_my_ga)->evaluatePopulation(*(stdev_my_pop));
    cout << "Initial Population\n";
    stdev_my_pop->sortedPrintOn(cout);
    cout << endl;
    outputBest(p->outputBestIndi,p);
  }
  else {
    continueES = (stdev_my_ga)->evaluateAndReplacePopulation(*(stdev_my_pop), stdev_my_offspring);
    outputBest(p->outputBestIndi,p);
  }
  
  if (continueES) {

    // generate a new offspring..
    stdev_my_ga->getOffspring(*(stdev_my_pop), stdev_my_offspring);
    //compareParentOffspring(); //not used now
    populationWithFitnessStdev();
  }
  else {
    printf("Final Population\n");
    stdev_my_pop->sortedPrintOn(cout);
    cout << endl;
  }

  firstEvaluation = false;
  return continueES;
}

//run the evolutionary algorithm
template<class EOT> 
bool runAlgorithmFull(EOT, ParamTuning* p)
{
  printGeneration();
  bool continueES = true;

  if (firstEvaluation) {
    // evaluate the parent population
    (full_my_ga)->evaluatePopulation(*(full_my_pop));
    cout << "Initial Population\n";
    full_my_pop->sortedPrintOn(cout);
    cout << endl;
    outputBest(p->outputBestIndi,p);
  }
  else {
    continueES = (full_my_ga)->evaluateAndReplacePopulation(*(full_my_pop), full_my_offspring);
    outputBest(p->outputBestIndi,p);
  }
  
  if (continueES) {
    // generate a new offspring..
    full_my_ga->getOffspring(*(full_my_pop), full_my_offspring);
    //compareParentOffspring(); //not used now
    populationWithFitnessFull();
  }
  else {
    printf("Final Population\n");
    full_my_pop->sortedPrintOn(cout);
    cout << endl;
  }

  firstEvaluation = false;
  return continueES;
}



bool ParamTuning::runEA()
{
  // choose correct algorithm version
  if(runReal == true)
    {
      return runAlgorithmReal(eoReal<FitT> (), this);
    }
  else if(runSimple == true)
    {
      return runAlgorithmSimple(eoEsSimple<FitT> (), this);
    }
  else if(runStdev == true)
    {
      return runAlgorithmStdev(eoEsStdev<FitT> (), this);
    }
  else
    {
      return runAlgorithmFull(eoEsFull<FitT> (), this);
    }
}

void ParamTuning::updateParameters()
{
  if (parameterUpdated)
    return;
  
  char args[100];
  string objectBounds;
  
  t_paramMap::iterator it;
  int id = 0;
  //paramList is a private member of type t_paramMap
  for (it = paramList.begin(); it != paramList.end(); it++) {
    EO_Params* tmp = it->second;
    tmp->id = id++;
    //keep appending these bounds to string args
    sprintf(args, "[%f,%f]", tmp->min, tmp->max);
    objectBounds.append(args);
  }
  
  //convert string paramFile to a c-string for input
  string argParamFile("tmp");
  //for input read the original paramFile
  ifstream inp_file ( paramFile.c_str(), ios::in );
  //in case we are pointing to a file in a directory
  //tokenize to /
  string dirName;
  string baseName;
  char* token;
  char charParamFile[300];
  const char* tempString = paramFile.c_str();
  memcpy(charParamFile,tempString,sizeof(charParamFile));
  printf("charParamFile = %s\n", charParamFile);
  
  token = strtok(charParamFile,"/");
  printf("token before while loop = %s\n", token);
  //token = strtok(paramFile.c_str(),"/");
  /* dirName = token; */
  /* printf("dirName = %s\n",dirName.c_str()); */
  /* baseName = dirName; */
  /* printf("baseName = %s\n", baseName.c_str()); */
  while(token != NULL){
    printf("token in while loop = %s\n",token);
    printf("baseName in while loop = %s\n",baseName.c_str() );
    printf("dirName in the while loop = %s\n",dirName.c_str());
    baseName = token;
    if(dirName.empty())
      dirName = dirName + baseName;
    else
      dirName = dirName + "/" + baseName;
    
    token = strtok(NULL,"/");
  }
  
  if(charParamFile[0] == '/')
    dirName = "/" + dirName;

  printf("final state of token = %s\n",token);
  printf("final state of baseName = %s\n", baseName.c_str());
  printf("final state of dirName = %s\n", dirName.c_str());
  
  //argParamFile = tmpGPU_SNN_ESEA.param
  //argParamFile = argParamFile + paramFile; //WORKS
  argParamFile = argParamFile + baseName;
  printf("argParamFile = %s\n",argParamFile.c_str());
  argParamFile = dirName = argParamFile;
  printf("argParamFile = %s\n",argParamFile.c_str());
  //write output to tmpGPU_SNN_ESEA.param
  //convert this into a c-string
  ofstream a_file (argParamFile.c_str());
  char ch;
  //copy everything from input file (MY_PARAM_FILE.param) to output file
  //tmpMY_PARAM_FILE.param, character by character.
  while ( !inp_file.eof()) {
    inp_file.get(ch);
    a_file.put(ch);
  }

  assert( vecSize == paramList.size());
  cout << "Population Size is : " << popSize << endl;
  cout << "Vector size is     : " << vecSize << endl;
  
  //Add these options to the output file
  //a_file << "--maxGen=1" << endl;
  
  a_file << "--vecSize=" << vecSize << endl;
  a_file << "--popSize=" << popSize << endl;
  a_file << "--objectBounds=" << objectBounds << "#Bounds for the given Object " << endl;
  a_file << "--initBounds=" << objectBounds   << "#Bounds for the init of the given Object " << endl;
    
  a_file.close();
  inp_file.close(); 

  //this is the last part that is tricky to understand
  char* argParam[2];
  string argParam0;
  argParam0 = "UCI-EO";
  argParam[0] = strdup(argParam0.c_str());
  argParamFile = "@" + argParamFile;
  argParam[1] = strdup(argParamFile.c_str());
   
  parser = new eoParser(2, argParam);
  state  = new eoState();
  
  assert( parser != NULL );

  // Run the appropriate algorithm (From four genotype options)
  if (runReal == true)
  {
    cout << "Using eoReal" << endl;
    setAlgorithmReal(eoReal<FitT> (), *parser, *state, this);
  }
  else if (runSimple == true)
  {
    cout << "Using eoEsSimple" << endl;
    setAlgorithmSimple(eoEsSimple<FitT> (), *parser, *state, this);
  }
  else if (runStdev == true)
  {
    cout << "Using eoEsStdev" << endl;
    setAlgorithmStdev(eoEsStdev<FitT> (), *parser, *state, this);
  }
  else
  {
    cout << "Using eoEsFull" << endl;
    setAlgorithmFull(eoEsFull<FitT> (), *parser, *state, this);
  }

  parameterUpdated=true;

  return;
}


int ParamTuning::getVectorSize()
{
  return vecSize;
}

int ParamTuning::getMaxGen()
{
  return maxGen;
}

void ParamTuning::addParam(string strName, double _min, double _max)
{
  //EO_Params eop;
  if(real_my_pop != NULL || simple_my_pop != NULL || stdev_my_pop != NULL || full_my_pop != NULL) {
    fprintf(stderr, "EO object already created..; Adding parameter %s pretty late\n", strName.c_str());
    return;
  }
  //this is the way you add stuff to maps.
  //from here you can tell that the key is a string and the
  //value is a pointer to a EO_Params object
  //it's also dynamically allocated so it's on the heap. 
  paramList[strName]=new EO_Params(_min,_max);  
      
  vecSize++;
}

double ParamTuning::getParamReal(uint32_t _IndiId, string _paramName)
{
  // first time we are getting parameters...
  // its time to generated the required EO objects..
  if(real_my_pop != NULL) 
    {
      updateParameters();
    }

  if (paramList.count(_paramName) == 0) {
    fprintf(stderr, "Unknown key value (%s) entered...\n", _paramName.c_str());
    return 0.0;
  }

  EO_Params* eop = paramList[_paramName];
  assert(eop != NULL);
  int id = eop->id;
  double retVal;
  //check to make sure _IndiId is within the bounds of the size of the population
  if (firstEvaluation) {
    if (!(_IndiId < (*real_my_pop).size())) {
      fprintf(stderr, "Requested IndiId = %d, size of actual population is %zu\n", _IndiId, (*real_my_pop).size());
      assert(_IndiId < (*real_my_pop).size());
    }
    retVal = (*real_my_pop)[_IndiId][id];
  }
  else {
    assert(_IndiId < real_my_offspring.size());
    retVal = real_my_offspring[_IndiId][id];
  }

  return retVal;
}

double ParamTuning::getParamSimple(uint32_t _IndiId, string _paramName)
{
  // first time we are getting parameters...
  // its time to generated the required EO objects..
  if(simple_my_pop != NULL) 
    {
      updateParameters();
    }

  if (paramList.count(_paramName) == 0) {
    fprintf(stderr, "Unknown key value (%s) entered...\n", _paramName.c_str());
    return 0.0;
  }

  EO_Params* eop = paramList[_paramName];
  assert(eop != NULL);
  int id = eop->id;
  double retVal;
  //check to make sure _IndiId is within the bounds of the size of the population
  if (firstEvaluation) {
    if (!(_IndiId < (*simple_my_pop).size())) {
      fprintf(stderr, "Requested IndiId = %d, size of actual population is %zu\n", _IndiId, (*simple_my_pop).size());
      assert(_IndiId < (*simple_my_pop).size());
    }
    retVal = (*simple_my_pop)[_IndiId][id];
  }
  else {
    assert(_IndiId < simple_my_offspring.size());
    retVal = simple_my_offspring[_IndiId][id];
  }

  return retVal;
}

double ParamTuning::getParamStdev(uint32_t _IndiId, string _paramName)
{
  // first time we are getting parameters...
  // its time to generated the required EO objects..
  if(stdev_my_pop != NULL) 
    {
      updateParameters();
    }

  if (paramList.count(_paramName) == 0) {
    fprintf(stderr, "Unknown key value (%s) entered...\n", _paramName.c_str());
    return 0.0;
  }

  EO_Params* eop = paramList[_paramName];
  assert(eop != NULL);
  int id = eop->id;
  double retVal;

  //check to make sure _IndiId is within the bounds of the size of the population
  if (firstEvaluation) {
    if (!(_IndiId < (*stdev_my_pop).size())) {
      fprintf(stderr, "Requested IndiId = %d, size of actual population is %zu\n", _IndiId, (*stdev_my_pop).size());
      assert(_IndiId < (*stdev_my_pop).size());
    }
    retVal = (*stdev_my_pop)[_IndiId][id];
  }
  else {
    assert(_IndiId < stdev_my_offspring.size());
    retVal = stdev_my_offspring[_IndiId][id];
  }

  return retVal;
}

double ParamTuning::getParamFull(uint32_t _IndiId, string _paramName)
{
  // first time we are getting parameters...
  // its time to generated the required EO objects..
  if(full_my_pop != NULL) 
    {
      updateParameters();
    }

  if (paramList.count(_paramName) == 0) {
    fprintf(stderr, "Unknown key value (%s) entered...\n", _paramName.c_str());
    return 0.0;
  }

  EO_Params* eop = paramList[_paramName];
  assert(eop != NULL);
  int id = eop->id;
  double retVal;
  //check to make sure _IndiId is within the bounds of the size of the population
  if (firstEvaluation) {
    if (!(_IndiId < (*full_my_pop).size())) {
      fprintf(stderr, "Requested IndiId = %d, size of actual population is %zu\n", _IndiId, (*full_my_pop).size());
      assert(_IndiId < (*full_my_pop).size());
    }
    retVal = (*full_my_pop)[_IndiId][id];
  }
  else {
    assert(_IndiId < full_my_offspring.size());
    retVal = full_my_offspring[_IndiId][id];
  }

  return retVal;
}

double ParamTuning::getParam(uint32_t _IndiId, string _paramName)
{
  if(runReal == true)
    {
      return getParamReal(_IndiId, _paramName);
    }
  else if(runSimple == true)
    {
      return getParamSimple(_IndiId, _paramName);
    }
  else if(runStdev == true)
    {
      return getParamStdev(_IndiId, _paramName);
    }
  else
    {
      return getParamFull(_IndiId, _paramName);
    }
}

bool ParamTuning::fitnessAlreadyExists(uint32_t _IndiId)
{
  if(firstEvaluation)
    return false;
  else {
    if (fitnessExists[_IndiId])
      {
	if(outputToFile)
	  {
	    fpEOCopy=fopen(outputFile.c_str(),"a");
	    fprintf(fpEOCopy, "[%d] Fitness already exists\n", _IndiId);
	    fclose(fpEOCopy);
	  }
      }
    return fitnessExists[_IndiId];
  }
}

// set the fitness value for population IndiId.  This works if you only assign fitness in serial.
void ParamTuning::setFitness (double _fitness, uint32_t _IndiId)
{
  fitnessValue[_IndiId]=_fitness;
  return;
}

// new setFitness function that allows an array of fitness to be set instead
// of a single value.  Of course you can pass single values by passing an array
// with just a single value.  This works for parallel fitness assignments
void ParamTuning::setFitness (double* _fitnessArray, uint32_t _IndiId, int _numConfig)
{
  for (int c=0; c < _numConfig; c++) {
    fprintf(stderr, "ParamTuning: Fitness value for population %u => %f\n", (_IndiId-_numConfig+c), _fitnessArray[c]);
  
    fitnessValue[_IndiId-_numConfig+c] = _fitnessArray[c];
  }
}

uint32_t ParamTuning::getPopulationSize()
{
  if (firstEvaluation) {
    return popSize;
  }
  else {
    
    if(runReal == true)
      {
	 return real_my_offspring.size();
      }
    else if(runSimple == true)
      {
	return simple_my_offspring.size();
      }
    else if(runStdev == true)
      {
	return stdev_my_offspring.size();
      }
    else
      {
	return full_my_offspring.size();
      }
   
  }
}

void ParamTuning::printSortedPopulation()
{
  if(runReal == true)
    {
      real_my_pop->sortedPrintOn(cout);
      cout << endl;
    }
  else if(runSimple == true)
    {
      simple_my_pop->sortedPrintOn(cout);
      cout << endl;
    }
  else if(runStdev == true)
    {
      stdev_my_pop->sortedPrintOn(cout);
      cout << endl;
    }
  else
    {
      full_my_pop->sortedPrintOn(cout);
      cout << endl;
    }
  
  return;
}



#endif
