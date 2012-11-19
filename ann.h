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

  std::vector<double> forwardProp(std::vector<double> lineIn);
  std::vector<double> findErrorVector(std::vector<double> output, int trainer);

  void backProp(std::vector<double> lineIn, int trainer);
  
  //these functions are only used by NeuralNetwork
 private: 
  double sigma(double weightedSum);
  double sigmaPrime(double weightedSum);

 private:
  
  layer hidden, output;

  double * inputs;
};
