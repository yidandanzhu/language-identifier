#	Language identifier: 

##	using neural network with one hidden layer to Perdict the probable language of each sentence. 

Input: five sequential characters from the text
Output: softmax over three options determining whether the language is English, French, or Italian.


##	Part One: Perdict the probable language of each sentence.


##	Part Two: Hyperparameter Optimization, use the dev data to optimize your hyperparameters

A list of the sets of hyperparameters that you tried on the dev data: [50,0.1],[50,0.05],[100,0.05],[120,0.05],[120,0.1]

#	The program should be run using a command like this:

python languageIdentification.py languageIdentification.data/train
languageIdentification.data/dev languageIdentification.data/test
