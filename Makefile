TESTS=$(wildcard tests/[^base]*.py)

.PHONY: test
tests:	
	@- $(foreach TEST,$(TESTS), \
			echo === Running test: $(TEST); \
			python -m unittest $(TEST); \
		)

test_scraper:
	python -m unittest tests/roster_scraper.py

data:
	mkdir ./data
	wget -O ./data http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz
	tar -xzf ./data/rotten_imdb.tar.gz ./data/rotten_imdb
