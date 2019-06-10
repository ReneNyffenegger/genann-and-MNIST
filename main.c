#include "genann.h"

#define USE_MNIST_LOADER
#define     MNIST_DOUBLE
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

// printf("%d\n", training_set[0].label);
// for (int y = 0; y<28; y++) {
//   for (int x = 0; x<28; x++) {
//     printf("%c", training_set[0].data[y][x] > 250 ? 'X': ' ');
//   }
//   printf("\n");
// }

   genann *nn = genann_init(28*28, 2, 28*28, 10);

   double desired_outputs[10];
// printf("size: %d\n", sizeof(desired_outputs));
//     exit(0);
   for (int train=0; train < cnt_training_set; train++) {
      #ifdef __STDC_IEC_559__ 
         memset(desired_outputs, 0, sizeof(desired_outputs));
      #else
         #error "expected __STDC_IEC_559__ not defined"
      #endif
      desired_outputs[training_set[train].label] = 1.0;
      genann_train(nn, (const double*) training_set[train].data, desired_outputs, 0.05);
      printf("Train %5d\n", train);
   }

   for (int test=0; test < cnt_test_set; test++) {

      double const *prediction = genann_run(nn, (const double*) test_set[test].data);

      double max = prediction[0];
      int ppp = 0;

      for (int i = 1; i<10; i++) {

        if (max < prediction[i]) {
          max = prediction[i];
          ppp = i;
        }

      }

      printf("test %5d, %d - %d\n", test, test_set[test].label, ppp);

   }

   genann_free(nn);

}
