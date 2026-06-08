.PHONY: all hw f8 mx clean

all: hw

hw:
	$(MAKE) -C hw

f8:
	$(MAKE) -C hw f8

mx:
	$(MAKE) -C hw mx

clean:
	$(MAKE) -C hw clean
