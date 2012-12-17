#include "data.h"


using namespace std;


Data::Data() :
  dataArray(vector<vector<double> >(NUM_EXAMPLES, vector<double>(NUM_ATTRIBUTES))),
  training(vector<int>(NUM_TRAIN)),
  validation(vector<int>(NUM_VALIDATE)),
  test(vector<int>(NUM_TEST)),
  coverType(vector<int>(NUM_EXAMPLES))
{
}

// read the data set in from the file. Instances in the file are 54 comma separated values per instance. Instance are separated by a new line character
void Data::readData() {


  ifstream dataFile;
  dataFile.open("covtype.data");


  string line;
  int outer = 0, inner = 0;
  double max[NUM_ATTRIBUTES], min[NUM_ATTRIBUTES];
  for (int i = 0; i < NUM_ATTRIBUTES; i++) {
    max[i] = 0;
    min[i] = 100000;
  }

  while (std::getline(dataFile, line)) {
    
    istringstream stream(line);
    char value[100];
    inner = 0;
   
    while (stream.getline(value, 100, ',')) {
      double number = std::atof(value);
      if (inner < NUM_ATTRIBUTES) {
       	dataArray[outer][inner] = number;
      	if (number > max[inner])
      	  max[inner] = number;
      	else if (number < min[inner])
      	  min[inner] = number;
 
      }
      else
      	coverType[outer] = number;

      inner++;
    }
    outer++;
  }

  //we will create the test and validation and test sets at the same time
  //in the following way: the training, validation, and test arrays will
  //contain the index in the dataArray of the desired example. Since there
  //is no order in the examples of the dataArray, we will construct the training
  //set by taking the first 1500 of each coverType, the validation by taking the 
  //next 500 of each coverType, and the test set will be all the rest.
  int numInTrain[7] = {0, 0, 0, 0, 0, 0, 0}, numInVal[7] = {0, 0, 0, 0, 0, 0, 0};
  int trainAmount = 0, valAmount = 0, testAmount = 0;
  for (int i = 0; i < NUM_EXAMPLES; i++) {
    for (int j = 0; j < NUM_ATTRIBUTES; j++) {
      dataArray[i][j] = (dataArray[i][j] - min[j])/(max[j] - min[j]);
      }
    if (numInTrain[coverType[i] - 1] < 1500 && trainAmount < NUM_TRAIN) {
      training[trainAmount] = i;
      trainAmount++;
    } 
    else if (numInVal[coverType[i] - 1] < 500 && valAmount < NUM_VALIDATE) {
      validation[valAmount] = i;
      valAmount++;
    } 
    else {
      test[testAmount] = i;
      testAmount++;
    }
  }


  dataFile.close(); //we no longer need the file to be open.

}
