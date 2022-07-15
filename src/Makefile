cc = gcc

sources = sparse_prims.c cpu_poisson.c cpu_ops.c common.c cpu_matrix_free_ops.c
objects = $(addsuffix .o, $(basename $(sources)))

flags = -g -std=c11 -fopenmp

target = cpu_poisson

all: $(target) 

$(target) : $(objects)
	$(cc) $(flags) -o $(target) $(objects) -lm

%.o : %.c
	$(cc) -c $(flags) $< -o $@

clean:
	rm -rf $(target) $(objects)


