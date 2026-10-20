JOB = mixed
SRC = main.tex preamble.tex $(wildcard chapters/*.tex)

$(JOB).pdf: $(SRC)
	pdflatex -interaction=nonstopmode -file-line-error -jobname=$(JOB) main.tex
	pdflatex -interaction=nonstopmode -file-line-error -jobname=$(JOB) main.tex

# 6-inch e-ink edition: same sources, compact geometry, selected by
# defining \KindleEdition before the class is read.
kindle:
	pdflatex -interaction=nonstopmode -file-line-error -jobname=$(JOB)-kindle "\def\KindleEdition{}\input{main.tex}"
	pdflatex -interaction=nonstopmode -file-line-error -jobname=$(JOB)-kindle "\def\KindleEdition{}\input{main.tex}"

clean:
	rm -f $(JOB).aux $(JOB)-kindle.aux $(JOB)-kindle.log $(JOB)-kindle.out $(JOB)-kindle.toc $(JOB).log $(JOB).out $(JOB).toc chapters/*.aux

.PHONY: clean kindle
