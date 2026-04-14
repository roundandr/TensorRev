.PHONY: all hw f8 clean

all: hw

hw:
	$(MAKE) -C hw

f8:
	$(MAKE) -C hw f8

clean:
	$(MAKE) -C hw clean
