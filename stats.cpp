#include "stats.h"


using namespace std;


Stats::Stats() {

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {

      statistics[i][j] = 0;

    }
  }
}



void Stats::print() {


  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      
      std::cout << statistics[i][j] << "\t";
    }
    std::cout << std::endl;
  }

  cout << endl;
  
}


int main() {

  Data *d = new Data();
  Stats *s = new Stats();

  cout << "reading data\n";
  d->readData();
  cout << "data read\nnow making net\n";
  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, 120);
  double alpha = 0.1; //this is the learning rate
  net->init(alpha);
  cout << "net initialized with random weights\n";
  
  const vector<int> train = d->getTraining();
  const vector<int> val = d->getValidation();
  const vector<int> test = d->getTest();

  //this is trainingish
  double meanErr = 10000, prevMeanErr = meanErr;
  int epoch = 0;
  double tolerance = 0.1; // I think it should be one std dev, but I havent calculated this yet.
  while (meanErr > 0.55) {
    for (int i = 0; i < NUM_TRAIN; i++) {
      const vector<double>& lineIn = d->getData()[train[i]];
      net->backProp(lineIn, d->getCover(train[i]));
    }
    
    prevMeanErr = meanErr;
    meanErr = 0;
    for (int i = 0; i < NUM_VALIDATE; i++) {
      const vector<double>& lineIn = d->getData()[val[i]];
      vector<double> thing = net->forwardProp(lineIn);
      vector<double> error = net->findErrorVector(thing, d->getCover(val[i]));
      double tempError = 0;
      for (int j = 0; j < error.size(); j++){
	tempError += pow(error[j], 2);
      }
      // tempError /= error.size();
      meanErr += tempError;
    }  
    meanErr /= NUM_VALIDATE;
    cout << epoch << " " << meanErr <<endl;
    epoch++;
  }


  //now we test
  double correct = 0;
  for (int i = 0; i < NUM_TEST; i++) {
    const vector<double>& lineIn = d->getData()[test[i]];
    vector<double> result = net->forwardProp(lineIn);
    //now to compare
    int maxIndex = 0;
    double maxVal = -1;
    for (int j = 0; j < 7; j++) {
      if (result[j] > maxVal){
	maxVal = result[j];
	maxIndex = j;
      }
    }
    int actualTree = d->getCover(test[i]);
    s->statistics[maxIndex][actualTree - 1]++;
    if (maxIndex == 0) correct++;
  }
  
  cout << "the percent correct is " << correct << endl;
  s->print();
  
}
