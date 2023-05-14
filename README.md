# NER-POS-models
In this project we implemented multiple level of models using pytorch for pos and ner classification.
tagger1:
simple mlp model with only single hidden layer, but using sophisticated scheduler for the learning rate, which 
allow us to play in training time in the learning rate.
tagger3:
we add pretrained words embedding - around 100,000 words, and merge them with all the words fromn the trainset,
we relate to words in the train set as parameters and we save them for the test inference.
tagger4:
we too add on the tagger3 the 3 characters of the suffix and prefix of each of the word in the train set, and we train the model.
tagger5:
we implemented much more sophisticated network, which relate to each word in two stage, character level and word level, in the 
character level we add to the vocab all the characters in the english language, and run conv1d layer on this sequence of embedding,
then we run max poling and concat all with the word embedding, then we feed it to fuclly connected layer.

some inisghts:
