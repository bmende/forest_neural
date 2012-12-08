#include "data.h"
#include <cmath>
#include <ctime>
#include <algorithm>

struct layer {
  
  int inputs, numNodes;
  double **weights;
  double *outputs;
  double *summed;
  double *primed;

};

class NeuralNetwork {

 public:

  NeuralNetwork(int numInputs, int numHidden);
  ~NeuralNetwork();

  void init(double alpha);

  std::vector<double> forwardProp(const std::vector<double>& lineIn);
  std::vector<double> findErrorVector(std::vector<double> trainee, int trainer);

  void backProp(const std::vector<double>& lineIn, int trainer);
  
  //these functions are only used by NeuralNetwork
 private: 
  double sigma(double weightedSum);
  double sigmaPrime(double weightedSum);

 private:
  double learnRate;

  layer hidden, output;

  double * inputs;
};
