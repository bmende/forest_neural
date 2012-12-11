#include "ann.h"


#define NUM_EPOCHS 300

class Stats {

 public:
  Stats();
  ~Stats();


  void print();
  void getWeights(int numHidden, double learnRate);
  void testWeights(std::string weightFileName, int numHidden, double learnRate);
  void reset();
  void writeToFile(std::string path, int epoch, int numHidden, double mse);

 private:
  int statistics[7][7];
  Data *d;
};
