#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#pragma once

#define NUM_EXAMPLES 581012
#define NUM_ATTRIBUTES 54

typedef std::vector< std::vector< double > > two_d_vec;

class Data {

public:
  
  Data();
  ~Data();

  void readData();
  double getData(int i, int j) { return dataArray[i][j]; }
  two_d_vec getData() { return dataArray; }
  int getCover(int i) { return coverType[i]; }
  std::vector< int > getCover() { return coverType; }
  
private:
 
  two_d_vec dataArray;
  std::vector< int > coverType;
};
