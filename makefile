build: 
	nvcc -o gierer main.cu solver.cu -g -G -lm -lcufft

run:  
	./gierer 256 1 100 0.5 1 6 0.003 100000 12345

submit: 
	sbatch job.sh

main: 
	nvcc -O3 -c main.cu

solver: 
	nvcc -O3 -c solver.cu

link: 
	nvcc -O3 -o gierer main.o solver.o -lm -lcufft

valgrind: 
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --trace-children=yes ./gierer 8 1 100 0.5 1 6 0.003 100000 12345

clean: 
	-rm gierer
	-rm *.o
	-rm *.out
