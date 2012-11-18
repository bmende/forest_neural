#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#pragma once

#define NUM_EXAMPLES 581012
#define NUM_ATTRIBUTES 54

class Data {

public:
  
  Data();
  ~Data();

  void readData();
  double getData(int i, int j) { return dataArray[i][j]; }
  int getCover(int i) { return coverType[i]; }
  
private:
 
  std::vector< std::vector<double> > dataArray;
  std::vector< int > coverType;
};
