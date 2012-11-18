#include "data.h"
#include <cmath>

struct layer {
  
  int inputs, numNodes;
  double **weights;
  double *outputs;

};

class NeuralNetwork {

 public:

  NeuralNetwork(int numInputs, int numHidden);
  ~NeuralNetwork();

  void init();

  void forwardProp(std::vector<double> lineIn);
  
  //these functions are only used by NeuralNetwork
 private: 
  double sigma(double weightedSum);
  double sigmaPrime(double weightedSum);

 private:
  
  layer hidden, output;

  double * inputs;
};
