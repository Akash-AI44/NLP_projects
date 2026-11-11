import spacy  # pyright: ignore[reportMissingImports]
from spacy import displacy  # pyright: ignore[reportMissingImports]

# Load the pre-trained English model into an "nlp" object
nlp = spacy.load("en_core_web_sm")

text = """
Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California
in 1976. On October 4, 2023, the company announced a new product at an
event in New York.
"""

# Run the text through the pipeline — this does tokenizing, tagging,
# and entity recognition all in one step
doc = nlp(text)

# Loop through detected entities
for ent in doc.ents:
    print(f"{ent.text:25} -> {ent.label_}  ({spacy.explain(ent.label_)})")

displacy.serve(doc, style="ent")
