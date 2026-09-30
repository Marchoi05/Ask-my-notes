# Ask My Notes

A chatbot that answers questions using only my own notes, and admits
when the answer isn't there. Runs fully free on my laptop.

## What problem does it solve?
[1-2 sentences: e.g. "I wanted an AI that answers from my notes
instead of guessing from the internet."]

## How it works
1. Splits my notes into paragraphs
2. Turns each paragraph into numbers that represent its meaning
3. Finds the paragraphs closest in meaning to my question
4. Gives them to a small local AI model (Llama 3.2 3B via Ollama),
   told to answer using only those paragraphs

## Results
- Search test: 20/20 questions found the correct paragraph
  (tested with top-3 and top-1 results)
- "Not in the notes" test: [X] out of [X] off-topic questions
  correctly refused, including borderline ones like Mars vs. Venus

## Limitations
- The test set is small (20 questions), so it can't prove the
  system is reliable in general
- Tested on short notes with very different topics; longer or
  more similar notes would be harder
- A small model can still make mistakes I haven't found yet

