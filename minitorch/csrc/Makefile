all:
	g++ -Wall -Wextra -std=c++14 -fPIC -c tensor.cpp -o tensor.o
	g++ --shared -o ../libtensor.so tensor.o

clean:
	rm -f *.o