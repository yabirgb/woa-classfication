
SEEDS = 37 42 440 699 752
BIN = ./target/release/woa-classification	
DATA = iris rand newthyroid

woa-pool:
	for seed in $(SEEDS); do \
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_10.const 3 $$seed 1 woa-pool 25 5 ;\
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_20.const 3 $$seed 1 woa-pool 25 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_10.const  8 $$seed 1 woa-pool 25 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_20.const 8 $$seed 1 woa-pool 25 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_10.const  3 $$seed 1 woa-pool 25 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_20.const 3 $$seed 1 woa-pool 25 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_10.const 3 $$seed 1 woa-pool 25 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_20.const 3 $$seed 1 woa-pool 25 5 ;\
	done 

woa-pool-30:
	for seed in $(SEEDS); do \
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_10.const 3 $$seed 1 woa-pool 30 5 ;\
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_20.const 3 $$seed 1 woa-pool 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_10.const  8 $$seed 1 woa-pool 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_20.const 8 $$seed 1 woa-pool 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_10.const  3 $$seed 1 woa-pool 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_20.const 3 $$seed 1 woa-pool 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_10.const 3 $$seed 1 woa-pool 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_20.const 3 $$seed 1 woa-pool 30 5 ;\
	done 

woa:
	for seed in $(SEEDS); do \
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_10.const 3 $$seed 1 woa 30 5 ;\
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_20.const 3 $$seed 1 woa 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_10.const  8 $$seed 1 woa 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_20.const 8 $$seed 1 woa 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_10.const  3 $$seed 1 woa 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_20.const 3 $$seed 1 woa 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_10.const 3 $$seed 1 woa 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_20.const 3 $$seed 1 woa 30 5 ;\
	done 

woa-ls:
	for seed in $(SEEDS); do \
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_10.const 3 $$seed 1 woa-ls 30 5 ;\
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_20.const 3 $$seed 1 woa-ls 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_10.const  8 $$seed 1 woa-ls 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_20.const 8 $$seed 1 woa-ls 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_10.const  3 $$seed 1 woa-ls 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_20.const 3 $$seed 1 woa-ls 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_10.const 3 $$seed 1 woa-ls 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_20.const 3 $$seed 1 woa-ls 30 5 ;\
	done 

woa-shake:
	for seed in $(SEEDS); do \
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_10.const 3 $$seed 1 woa-shake 30 5 ;\
		$(BIN) ./data/iris_set.dat ./data/iris_set_const_20.const 3 $$seed 1 woa-shake 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_10.const  8 $$seed 1 woa-shake 30 5 ;\
		$(BIN) ./data/ecoli_set.dat ./data/ecoli_set_const_20.const 8 $$seed 1 woa-shake 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_10.const  3 $$seed 1 woa-shake 30 5 ;\
		$(BIN) ./data/rand_set.dat ./data/rand_set_const_20.const 3 $$seed 1 woa-shake 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_10.const 3 $$seed 1 woa-shake 30 5 ;\
		$(BIN) ./data/newthyroid_set.dat ./data/newthyroid_set_const_20.const 3 $$seed 1 woa-shake 30 5 ;\
	done 