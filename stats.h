#include "ann.h"




class Stats {

 public:
  Stats();
  ~Stats();


  void print();
  void getWeights(int numHidden, double learnRate);
  void testWeights(std::string weightFileName, int numHidden, double learnRate);
  void reset();
  void writeToFile(std::string path);

  int statistics[7][7];
};
