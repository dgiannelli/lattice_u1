.DEFAULT_GOAL := install

install:
	mkdir -p build
	cd build &&	cmake -DCMAKE_PREFIX_PATH=$(CONDA_PREFIX) ..
	$(MAKE) -C build

clean:
	rm -r build/

