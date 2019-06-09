#include "genann.h"

#define USE_MNIST_LOADER
#include "mnist.h"


int main() {

   srand(2808);

   unsigned int cnt_training_set, cnt_test_set;
   mnist_data      *training_set,    *test_set;
   
   int ret;
   if (ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &training_set, &cnt_training_set)) {
       printf("Could not load training set (%d)\n", ret);
       exit(1);
   }

   if (ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_set, &cnt_test_set)) {
       printf("Could not load test set (%d)\n", ret);
       exit(1);
   }
   printf("count training set: %d, count test set: %d\n", cnt_training_set, cnt_test_set);

   printf("%d\n", training_set[0].label);
   for (int y = 0; y<28; y++) {
     for (int x = 0; x<28; x++) {
       printf("%c", training_set[0].data[y][x] > 250 ? 'X': ' ');
     }
     printf("\n");
   }

}
