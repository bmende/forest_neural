#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#pragma once


class Data {

public:
  
  Data();
  ~Data();

  void readData();
  
private:
 
  double dataArray[581012][54];
  int coverType[581012];
};
