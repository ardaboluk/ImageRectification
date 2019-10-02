CC = g++ # C compiler
INCLUDEPATH = 
LIBRARYPATH = 
LIBS = `pkg-config --cflags --libs opencv`
CFLAGS = $(INCLUDEPATH) $(LIBRARYPATH) -Wall -O2 -g # C flags
LDFLAGS = # linking flags
RM = rm -f  # rm command
TARGET = ImageRectification # target

SRCS =  correspondenceChooser.cpp estimator.cpp main.cpp preprocessing.cpp rectification.cpp util.cpp # source files

.PHONY: all
all: ${TARGET}

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

.PHONY: clean
clean:
	-${RM} ${TARGET}