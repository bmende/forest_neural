#include "ann.h"


using namespace std;

NeuralNetwork::NeuralNetwork(int numInputs, int numHidden) {


  hidden.inputs = numInputs; hidden.numNodes = numHidden;
  hidden.weights = new double*[hidden.numNodes];
  for (int i = 0; i < hidden.numNodes; i++) 
    hidden.weights[i] = new double[hidden.inputs];
  hidden.outputs = new double[hidden.numNodes];

  output.inputs = hidden.numNodes; output.numNodes = 7;
  output.weights = new double*[output.numNodes];
  for (int i = 0; i < output.numNodes; i++)
    output.weights[i] = new double[output.inputs];
  output.outputs = new double[output.numNodes];

  inputs = new double[numInputs];

}

void NeuralNetwork::init() {

  //initializing weights to 0.5 each. TODO: make random initial weights

  for (int i = 0; i < hidden.numNodes; i++) {
    for (int j = 0; j < hidden.inputs; j++) {
      hidden.weights[i][j] = 0.5;
    }
    hidden.outputs[i] = 0;
  }

  for (int i = 0; i < output.numNodes; i++) {
    for (int j = 0; j < output.inputs; j++) {
      output.weights[i][j] = 0.5;
    }
    output.outputs[i] = 0;
  }
}

double NeuralNetwork::sigma(double weightedSum) {
  //this function is sigma(x) = 1/(1+e^(-x))

  weightedSum *= -1;
  double answer = 1 + exp(weightedSum);
  answer = 1/answer;

  return answer;
}

double NeuralNetwork::sigmaPrime(double weightedSum) {
  double sig = sigma(weightedSum);

  return sig*(1-sig);
}

void NeuralNetwork::forwardProp(vector<double> lineIn) {
  
  if (lineIn.size() != 54) {
    cerr << "line in not correct size\n";
    exit(1);
  } 
  //putting data in input nodes
  for (int i = 0; i < lineIn.size(); i++) {
    inputs[i] = lineIn[i];
  }

  //calculating hidden node outputs.
  for (int node = 0; node < hidden.numNodes; node++) {
    double sum = 0;
    for (int in = 0; in < hidden.inputs; in++) {
      sum += (inputs[in] * hidden.weights[node][in]);
    }
    hidden.outputs[node] = sigma(sum);
  }
  
  //now for output node calculations
  for (int node = 0; node < output.numNodes; node++) {
    double sum = 0;
    for (int in = 0; in < output.inputs; in++) {
      sum += (hidden.outputs[in] * output.weights[node][in]);
    }
    output.outputs[node] = sigma(sum);
  }
  cout << endl;
}


int main() {

  Data *d = new Data();

  cout << "reading data\n";
  d->readData();
  cout << "data read\nnow making net\n";
  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, 120);
  net->init();
  cout << "net initialized with weights = 0.5\n";
  
    net->forwardProp(d->getData()[0]);


}
