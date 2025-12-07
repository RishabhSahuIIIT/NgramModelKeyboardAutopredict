import re
from collections import defaultdict, Counter
import sys

class NgramCharacterModel:
    def __init__(self, corpus:str, n:int=2):
        # Initialize the class variables
        self.N = n # previous n characters that will be considered
        self.corpusSize = len(corpus)
        self.modelCounts = defaultdict(Counter)
        self.vocab = set()
        #print(f"Corpus size: {self.corpusSize}")
        if corpus:
            self._train(corpus)
        
        # Print debug information after training
        #print(f"Total unique contexts: {len(self.modelCounts)}")
        #print(f"Sample contexts: {list(self.modelCounts.keys())[:10]}")

    def _train(self, corpus:str):
        # Train the language model on the given corpus input
        #clean data
        cleanedCorpus = corpus.encode("ascii","ignore").decode()
        cleanedCorpus = re.sub(r"[\n\r\t\xa0]", " ", cleanedCorpus)
        cleanedCorpus = re.sub(r"\?", "", cleanedCorpus)
        cleanedCorpus = re.sub(r"[^\w\s.]", "", cleanedCorpus)
        cleanedCorpus = re.sub(r"[._]", "", cleanedCorpus) # remove fullstops and
        cleanedCorpus = re.sub(r"\d", "", cleanedCorpus) # remove numbers
        cleanedCorpus = re.sub(r"\s+", " ", cleanedCorpus).strip() # normalise spaces
        cleanedCorpus = cleanedCorpus.lower() # convert to lower case
        
        #print(f"Cleaned corpus length: {len(cleanedCorpus)}")
        
        words = cleanedCorpus.split(" ")
        #print(f"Total words: {len(words)}")
        #print(f"First 10 words: {words[:10]}")
        
        startPadding = '$' * (self.N - 1)
        endPadding = '$' * (self.N - 1)
        
        for word in words:
            self.vocab.add(word)
            paddedWord = f"{startPadding}{word}{endPadding}"
            
            for pos in range(len(paddedWord) - self.N + 1):
                context = paddedWord[pos:pos+self.N-1]
                nextChar = paddedWord[pos + self.N - 1]
                self.modelCounts[context][nextChar] += 1
        
        #print(f"Vocabulary size: {len(self.vocab)}")

    def predict_top_words(self, prefix, top_k=10):
        # Ensure prefix is not empty
        if not prefix:
            return []
        
        start_pad = '$' * (self.N-1)
        candidates = [(prefix, 1.0)]
        completed_words = []
        
        # Very low probability threshold to stop exploration
        MIN_PROB_THRESHOLD = 1e-6
        
        # Extended prediction with probability-based stopping
        iteration = 0
        while candidates and iteration < 20:
            iteration += 1
            new_candidates = []
            
            for cand, prob in candidates:
                # Skip if probability is too low
                if prob < MIN_PROB_THRESHOLD:
                    continue
                
                # Determine the context based on the candidate
                if len(cand) < self.N - 1:
                    context = (start_pad + cand)[-self.N+1:]
                else:
                    context = cand[-(self.N-1):]
                
                # Skip if no context
                if context not in self.modelCounts:
                    continue
                
                # Calculate probabilities for next characters
                total = sum(self.modelCounts[context].values())
                
                # Track unique candidates to prevent duplicates
                unique_candidates = set()
                
                for char, count in self.modelCounts[context].most_common():
                    new_prob = prob * (count / total)
                    
                    # Skip if probability is too low
                    if new_prob < MIN_PROB_THRESHOLD:
                        continue
                    
                    new_word = cand + char
                    
                    # Avoid duplicate candidates
                    if new_word not in unique_candidates:
                        unique_candidates.add(new_word)
                        
                        # Check if word is complete (has end marker)
                        if char == '$':
                            final_word = new_word.strip('$').rstrip('$')
                            # Only add complete words that start with the prefix
                            if final_word.startswith(prefix):
                                completed_words.append((final_word, new_prob))
                        else:
                            new_candidates.append((new_word, new_prob))
            
            # Update candidates for next iteration
            candidates = new_candidates
        
        # Combine and sort all completed words by probability
        all_predictions = completed_words
        
        # If no complete words found, use the best partial candidates
        if not all_predictions and candidates:
            all_predictions = candidates
        
        # Sort by probability and return top K unique words
        unique_predictions = []
        seen = set()
        for word, prob in sorted(all_predictions, key=lambda x: -x[1]):
            if word not in seen:
                unique_predictions.append(word)
                seen.add(word)
            
            if len(unique_predictions) == top_k:
                break
        
        return unique_predictions
        
    def _word_probability(self, word):
        # Calculates the probability of the word, based on the ngram probabilities
        # Add padding to handle n-gram context
        padded = '$'*(self.N-1) + word + '$'*(self.N-1)
        
        # Initialize probability
        prob = 1.0
        
        # Calculate probability for each character transition
        for i in range(len(padded) - self.N + 1):
            context = padded[i:i+self.N-1]
            next_char = padded[i+self.N-1]
            
            # If context or character not in model, return very low probability
            if context not in self.modelCounts or next_char not in self.modelCounts[context]:
                return 0.0001  # Small non-zero probability
            
            # Calculate conditional probability
            total = sum(self.modelCounts[context].values())
            char_prob = self.modelCounts[context][next_char] / total
            
            # Multiply probabilities (log would be better for longer words)
            prob *= char_prob
        
        return prob