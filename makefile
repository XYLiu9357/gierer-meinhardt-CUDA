# compiler settings
CC = nvcc -O2

all: main solver link

build: 
	nvcc -o gierer main.cu solver.cu -g -G -lm -lcufft

run:  
	./gierer 128 1 100 0.5 1 6 0.003 100000 12345

job: 
	sbatch --wait job.sh

main: 
	$(CC) -c main.cu

solver: 
	$(CC) -c solver.cu

link: 
	$(CC) -o gierer main.o solver.o -lm -lcufft

valgrind: 
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --trace-children=yes ./gierer 8 1 100 0.5 1 6 0.003 100000 12345

clean: 
	-rm gierer
	-rm *.o
	-rm *.out
