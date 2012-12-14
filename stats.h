#include "ann.h"

class Stats {

 public:
  Stats(int numEpochs, bool debug);
  ~Stats();


  void print();
  void getWeights(double learnRate);
  void testWeights(std::string weightFileName, int numHidden, double learnRate);
  void reset();
  void writeToFile(std::string path, int epoch, int numHidden, double mse);

 private:
  int statistics[7][7];
  Data *d;
  int num_epochs;
  bool isDebug;
};
