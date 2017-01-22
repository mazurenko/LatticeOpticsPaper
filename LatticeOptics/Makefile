# -*- mode: Makefile -*-
# Time-stamp: "2015-01-08 12:44:47 sb"

#  file       Makefile
#  copyright  (c) Sebastian Blatt 2015

project := latticelifetime
targets := $(project).pdf $(project).dvi
subs := figures
paper := letter #a4
pdfquality := default #screen
pdfpermissions := -4
# -4 allow all, then subtract
# 4 print document
# 8 modify contents
# 16 copy text and graphics
# 32 add/modify annotations

all : $(targets) $(subs)

%.pdf: %.ps
	@echo "ps2pdf: $< -> $@"
	@ps2pdf14 "-sPAPERSIZE=$(paper)" "-dPDFSETTINGS=/$(pdfquality)" "-dPermissions#$(pdfpermissions)" $<

%.ps: %.dvi
	@echo "dvips: $< -> $@"
	@dvips -t $(paper) -Ppdf -G0 $< -o $@

%.dvi: %.tex $(subs)
	@echo "latex: $< -> $@"
	@latex $<
	@if [ -f $(<:.tex=.bib) ]; then bibtex $(<:.tex=); fi
	@if [ grep '\printindex' $< ]; then \
			makeindex -s $(HOME)/elisp/latex/style/mkidx.ist -g $(<:.tex=.idx); fi
	@latex $<
	@latex $<
	@-rm $(foreach t,aux log blg ilg ind bbl idx loc soc,$(<:.tex=.$(t))) \
			$(project)Notes.bib

.PHONY: clean $(subs)

$(subs):
	@$(MAKE) -C $@ all

clean :
	@-rm $(targets)
	@for i in $(subs) ; do $(MAKE) -C $$i clean ; done
