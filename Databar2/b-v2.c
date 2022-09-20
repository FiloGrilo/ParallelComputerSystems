#include <stdio.h>

#define ITERATIONS 10000
#define ARRAY_SIZE 32768

int main(int argc, char** argv) {
  int data[ARRAY_SIZE];
  
  for(int i=0; i<ARRAY_SIZE; i++) {
    data[i] = i+1;
  }

  data[ARRAY_SIZE-1] = -1;

  for(int i=0; i<ITERATIONS; i++) {
    int sum = 0;
    int index = 0;
    while(index != ARRAY_SIZE) {
      sum += data[index];
      index++;
    }
  }

  return 0;
}