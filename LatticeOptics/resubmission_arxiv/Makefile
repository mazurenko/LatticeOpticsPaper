# -*- mode: Makefile -*-
# Time-stamp: "2018-06-22 12:09:34 sb"

#  file       Makefile
#  copyright  (c) Sebastian Blatt 2015, 2016, 2017, 2018

project := latticeoptics
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
	#@bibtex $(project)
	@pdflatex $<
	@pdflatex $<
	@-rm $(foreach t,aux log blg ilg ind bbl idx loc soc,$(<:.tex=.$(t))) \
			$(project)Notes.bib

diff:
	@echo "latexdiff ../submission_pr_applied/$(project).tex $(project).tex > $(project)_diff.tex"
	@latexdiff ../submission_pr_applied/$(project).tex  $(project).tex  > $(project)_diff.tex
	@pdflatex $(project)_diff.tex

.PHONY: clean diff $(subs)


clean :
	@-rm $(targets)
