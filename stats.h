#include "ann.h"




class Stats {

 public:
  Stats();
  ~Stats();


  void print();
    void writeToFile(std::string path);


  int statistics[7][7];
};
