#
# Makefile for mrmr feature selector.
# Currently set for heavy debugging
#

PYTHON_INCLUDE = -I/home/pgarst/tools/anaconda/include/python2.7 -I/home/pgarst/tools/anaconda/lib/python2.7/config
CC	= g++
LD	= g++
CFLAGS = -g -O0 -fpic -I. -I/usr/include $(PYTHON_INCLUDE)
LIBS	= 

TOP        = ../..
SWIGEXE    = swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
CXXSRCS    = mrdata.cpp
TARGET     = mrmrlib
LIBS       = -lm

all: build

OBJ = Obj/mrdata.o Obj/sort2.o Obj/nrutil.o Obj/pbetai.o Obj/DataTable.o Obj/mrmrmodule.o
COBJ = Obj/mrdata.o Obj/sort2.o Obj/nrutil.o Obj/pbetai.o Obj/DataTable.o

Obj/%.o:     %.cpp
	$(CC) -O $(CFLAGS) -c $(OUTPUT_OPTION) $<

Objg/%.o:       %.cpp
	$(CC) -g $(CFLAGS) -c $(OUTPUT_OPTION) $<

testmr: buildc Obj/testmr.o
	g++ -g -I/usr/include -L. -o testmr Obj/testmr.o ./cmrmrlib.so

buildc: $(COBJ) 
	g++ -shared  -I/usr/include $(COBJ) -o cmrmrlib.so

build: $(OBJ) 
	g++ -shared  -I/usr/include $(OBJ) -o _mrmrlib.so
	mv _mrmrlib.so build

clean:
	rm -rf __pycache__
	rm -f *_wrap* *~ .~* mypython *.pyc
	rm -f core 
	rm -f Obj/*.o Objg/*.o *.o *.so 
	rm -f $(TARGET).py
	if test -f $(SRCDIR)$(RUNME).py; then rm -f $(RUNME)3.py $(RUNME)3.py.bak; fi
	case "x$(SRCDIR)" in x|x./);; *) rm -f $(RUNME).py;; esac

