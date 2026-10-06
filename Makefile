CXX = g++

CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

EIGEN = -I/usr/include/eigen3

INCLUDES = -Iinclude

SRC = \
src/poisson.cpp \
src/preconditioners.cpp \
src/utilities.cpp \
src/iterative.cpp \
src/direct.cpp \
src/main.cpp

TARGET = poisson_solver

all:
	$(CXX) $(CXXFLAGS) $(SRC) \
	$(INCLUDES) $(EIGEN) \
	-o $(TARGET)

run:
	./$(TARGET)

clean:
	rm -f $(TARGET)