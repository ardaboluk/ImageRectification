CXX = g++ # C compiler
INCLUDEPATH = 
LIBRARYPATH = 
LIBS = `pkg-config --cflags --libs opencv`
CXXFLAGS = $(INCLUDEPATH) -Wall -O2 -g 
LDFLAGS = $(LIBRARYPATH) # linking flags
RM = rm -f  # rm command
TARGET = ImageRectification # target

SRCS = correspondenceChooser.cpp estimator.cpp main.cpp preprocessing.cpp rectification.cpp util.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: ${TARGET}

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $<

.PHONY: clean
clean:
	-${RM} ${OBJS} ${TARGET}