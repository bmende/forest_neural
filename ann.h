#include "data.h"


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

 private:
  
  layer hidden, output;

  double * inputs;
};
