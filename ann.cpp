#include "ann.h"


using namespace std;

NeuralNetwork::NeuralNetwork(int numInputs, int numHidden) {


  hidden.inputs = numInputs; hidden.numNodes = numHidden;
  hidden.weights = new double*[hidden.numNodes];
  for (int i = 0; i < hidden.numNodes; i++) 
    hidden.weights[i] = new double[hidden.inputs];
  hidden.outputs = new double[hidden.numNodes];

  output.inputs = hidden.numNodes; output.numNodes = 7;
  output.weights = new double*[output.inputs];
  for (int i = 0; i < output.inputs; i++)
    output.weights[i] = new double[output.numNodes];
  output.outputs = new double[output.numNodes];

}

void NeuralNetwork::init() {

  //initializing weights to 0.5 each. TODO: make random initial weights

  for (int i = 0; i < hidden.numNodes; i++) {
    for (int j = 0; j < hidden.inputs; j++) {
      cout << j << " " << i << endl;
      hidden.weights[j][i] = 0.5;
    }
    hidden.outputs[i] = 0;
  }
  cout << "safe\n";
  for (int i = 0; output.numNodes; i++) {
    for (int j = 0; j < output.inputs; j++) {
      cout << j << " " << i << endl;
      output.weights[i][j] = 0.5;
    }
    output.outputs[i] = 0;
  }
}


int main() {

  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, 120);
  net->init();
  Data *d = new Data();

  cout << "reading data\n";
  d->readData();
  cout << "data read\nnow making net\n";


  cout << "net initialized with weights = 0.5\n";

}
