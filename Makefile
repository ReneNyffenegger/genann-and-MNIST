genann-and-MNIST: main.c mnist.h genann.o
	gcc -O3 main.c genann.o -lm -o $@

genann.o: genann.c genann.h
	gcc -c -O3 $< -o $@
