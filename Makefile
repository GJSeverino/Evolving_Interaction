main: main.o CTRNN.o TSearch.o PerceptualCrosser.o random.o
	g++ -pthread -o main main.o CTRNN.o TSearch.o PerceptualCrosser.o random.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -pthread -c -O3 random.cpp
CTRNN.o: CTRNN.cpp random.h CTRNN.h
	g++ -pthread -c -O3 CTRNN.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -pthread -c -O3 TSearch.cpp
PerceptualCrosser.o: PerceptualCrosser.cpp PerceptualCrosser.h TSearch.h CTRNN.h random.h VectorMatrix.h
	g++ -pthread -c -O3 PerceptualCrosser.cpp
main.o: main.cpp CTRNN.h PerceptualCrosser.h TSearch.h
	g++ -pthread -c -O3 -std=c++11 main.cpp
clean:
	rm *.o main
