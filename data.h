#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#pragma once

#define NUM_EXAMPLES 581012 // total instances in data set
#define NUM_ATTRIBUTES 54 // number of input nodes
#define NUM_TRAIN 7*1620  // size of training set (1620 of each type)
#define NUM_VALIDATE 7*540 // size of validations et (540 of each type)
#define NUM_TEST (NUM_EXAMPLES - (NUM_TRAIN + NUM_VALIDATE)) // test set is all the rest

typedef std::vector< std::vector< double > > two_d_vec;

class Data {

public:
  
  Data();
  ~Data();

  void readData();
  double getData(int i, int j) { return dataArray[i][j]; }
  const two_d_vec& getData() { return dataArray; }
  const std::vector<int>& getTraining() { return training; }
  const std::vector<int>& getValidation() { return validation; }
  const std::vector<int>& getTest() { return test; }
  int getCover(int i) { return coverType[i]; }
  std::vector< int > getCover() { return coverType; }
  
private:
 
  two_d_vec dataArray;
  std::vector< int > coverType, training, validation, test;
};
