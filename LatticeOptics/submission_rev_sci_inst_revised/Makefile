# -*- mode: Makefile -*-
# Time-stamp: "2019-01-27 17:35:53 sb"

#  file       Makefile
#  copyright  (c) Sebastian Blatt 2015 -- 2019

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
	@bibtex $(project)
	@pdflatex $<
	@pdflatex $<
	@-rm $(foreach t,aux log blg ilg ind bbl idx loc soc,$(<:.tex=.$(t))) \
			$(project)Notes.bib

diff:
	@echo "latexdiff ../submission_rev_sci_inst/$(project).tex $(project).tex > $(project)_diff.tex"
	@latexdiff ../submission_pr_applied/$(project).tex  $(project).tex  > $(project)_diff.tex
	@pdflatex $(project)_diff.tex

.PHONY: clean diff $(subs)


clean :
	@-rm $(targets)
