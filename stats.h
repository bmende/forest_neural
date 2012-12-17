#include "ann.h"

class Stats {

 public:
  Stats(int numEpochs, bool debug);
  ~Stats();


  void print();
  void getWeights(int numHidden, double learnRate);
  void testWeights(std::string weightFileName, int numHidden, double learnRate);
  void reset();
  void writeToFile(std::string path, int epoch, int numHidden, double mse);

 private:
  // a table with the network's type classification on the y axis and the actual classification on the x axis. Each instance passed through the net increments the corresponding cell in this table. For example, if the network classified the input as type 4 but it was actually type 3, the element at [3][2] would be incremented (array is 0 indexed).
  int statistics[7][7];
  Data *d;
  int num_epochs;
  bool isDebug;
};
