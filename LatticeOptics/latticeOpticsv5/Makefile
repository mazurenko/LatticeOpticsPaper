# -*- mode: Makefile -*-
# Time-stamp: "2018-06-11 16:34:49 sb"

#  file       Makefile
#  copyright  (c) Sebastian Blatt 2015, 2016, 2017, 2018

project := LatticeOpticsV5
targets := $(project).pdf
subs := figures
paper := letter #a4
pdfquality := default #screen
pdfpermissions := -4
# -4 allow all, then subtract
# 4 print document
# 8 modify contents
# 16 copy text and graphics
# 32 add/modify annotations

all: $(targets) $(subs)


%.pdf: %.tex $(subs)
	@echo "pdflatex: $< -> $@"
	@pdflatex $<
	@bibtex $(project)
	@pdflatex $<
	@pdflatex $<
	@-rm $(foreach t,aux log blg ilg ind bbl idx loc soc,$(<:.tex=.$(t))) \
			$(project)Notes.bib

.PHONY: clean $(subs)


clean :
	@-rm $(targets)
