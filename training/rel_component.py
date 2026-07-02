# -*- coding: utf-8 -*-
"""
rel_component.py
════════════════
Custom component for Relation Extraction in SpaCy v3.
Registers the "relation_extractor" component and models to the SpaCy registry.
"""

from typing import List, Dict, Tuple, Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.training import Example
from thinc.api import Model, Config
from thinc.types import Floats2d
import spacy

# Register custom extension attribute on Doc to store predicted relations
if not Doc.has_extension("rel"):
    Doc.set_extension("rel", default={})

@Language.factory(
    "relation_extractor",
    assigns=["doc._.rel"],
    default_config={
        "threshold": 0.5,
        "model": {
            "@architectures": "rel_model.v1",
            "tok2vec": {
                "@architectures": "spacy.Tok2VecListener.v1",
                "width": 96,
                "upstream": "*"
            },
            "nO": None
        }
    }
)
def make_relation_extractor(nlp: Language, name: str, model: Model, threshold: float):
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)

class RelationExtractor:
    def __init__(self, vocab, model: Model, name: str = "relation_extractor", threshold: float = 0.5):
        self.vocab = vocab
        self.model = model
        self.name = name
        self.threshold = threshold

    def __call__(self, doc: Doc) -> Doc:
        # Predict relations for doc
        if len(doc.ents) < 2:
            doc._.rel = {}
            return doc
        
        # Format instances for prediction
        instances = []
        for i, ent1 in enumerate(doc.ents):
            for j, ent2 in enumerate(doc.ents):
                if i != j:
                    instances.append((ent1, ent2))
        
        if not instances:
            doc._.rel = {}
            return doc
            
        # Run model forward pass
        predictions = self.model.predict(instances)
        
        # Assign predicted relations to doc._.rel
        relations = {}
        for (ent1, ent2), pred in zip(instances, predictions):
            # pred is a list of floats (probabilities for each relation label)
            # Find the best class index
            best_idx = int(pred.argmax())
            prob = float(pred[best_idx])
            if prob >= self.threshold:
                label = self.model.attrs["labels"][best_idx]
                key = (ent1.start_char, ent1.end_char, ent2.start_char, ent2.end_char)
                relations[key] = {label: prob}
                
        doc._.rel = relations
        return doc

    def initialize(self, get_examples, nlp=None, labels=None):
        if labels is not None:
            self.model.attrs["labels"] = labels
        else:
            # Gather labels from training examples
            labels = set()
            for example in get_examples():
                for rel_dict in example.reference._.rel.values():
                    labels.update(rel_dict.keys())
            self.model.attrs["labels"] = sorted(list(labels))
            
        nO = len(self.model.attrs["labels"])
        
        # 1. Initialize tok2vec sub-model manually
        tok2vec = self.model.get_ref("tok2vec")
        nlp_dummy = spacy.blank("id")
        dummy_doc = nlp_dummy.make_doc("dummy text for setup")
        tok2vec.initialize(X=[dummy_doc])
        
        # 2. Initialize classifier sub-model manually with exact feature dimensions
        classifier = self.model.get_ref("classifier")
        nI = tok2vec.get_dim("nO") * 2 if tok2vec.has_dim("nO") else 96 * 2
        dummy_features = self.model.ops.alloc2f(1, nI)
        dummy_targets = self.model.ops.alloc2f(1, nO)
        classifier.initialize(X=dummy_features, Y=dummy_targets)
        
        # 3. Finally initialize the parent model wrapper
        self.model.initialize(X=self._get_dummy_instances(), Y=self.model.ops.alloc2f(1, nO))

    def _get_dummy_instances(self):
        # Create a dummy Doc with 2 entities for initialization
        nlp = spacy.blank("id")
        doc = nlp.make_doc("dummy text for setup")
        doc.ents = [Span(doc, 0, 1, label="DUMMY1"), Span(doc, 2, 3, label="DUMMY2")]
        return [(doc.ents[0], doc.ents[1])]

    def to_disk(self, path, exclude=tuple()):
        # Ensure the output directory exists before saving
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        # Save model and settings to disk
        self.model.to_disk(path / "model")
        
    def from_disk(self, path, exclude=tuple()):
        # Load model and settings from disk
        self.model.from_disk(path / "model")
        return self


# --- Neural Network Model Architecture for Relation Extraction ---

@spacy.registry.architectures("rel_model.v1")
def create_relation_model(tok2vec: Model[List[Doc], List[Floats2d]], nO: Optional[int] = None) -> Model:
    # A simple neural network model that calculates representation vectors for span pairs
    # and runs a classifier over them.
    from thinc.api import Model, chain, Relu, list2array, with_array
    
    # Custom forward pass
    def forward(model, instances, is_train):
        tok2vec_model = model.get_ref("tok2vec")
        # Extract documents from spans
        docs = [span1.doc for span1, span2 in instances]
        
        # Calculate representation vectors for each span pair (pooling span tokens)
        ops = model.ops
        
        if is_train:
            # Get token vectors and backprop function
            tok2vec_outputs, backprop_tok2vec = tok2vec_model.begin_update(docs)
            
            features = []
            for i, (span1, span2) in enumerate(instances):
                vectors = tok2vec_outputs[i]
                v1 = vectors[span1.start : span1.end].mean(axis=0)
                v2 = vectors[span2.start : span2.end].mean(axis=0)
                pair_vector = ops.xp.hstack([v1, v2])
                features.append(pair_vector)
                
            X = ops.asarray2f(features)
            
            # Classifier layer (Linear + Softmax)
            classifier = model.get_ref("classifier")
            Y, backprop_classifier = classifier.begin_update(X)
            
            def backprop(dY):
                # Backpropagation logic for classifier updates
                dX = backprop_classifier(dY)
                # We send zero gradient to tok2vec to prevent the relation extractor
                # task from interfering with or corrupting the shared tok2vec representations
                # which are heavily optimized by the NER component.
                dtok2vec = [ops.alloc2f(v.shape[0], v.shape[1]) for v in tok2vec_outputs]
                backprop_tok2vec(dtok2vec)
                return []
                
            return Y, backprop
        else:
            # Predict pass
            tok2vec_outputs = tok2vec_model.predict(docs)
            features = []
            for i, (span1, span2) in enumerate(instances):
                vectors = tok2vec_outputs[i]
                v1 = vectors[span1.start : span1.end].mean(axis=0)
                v2 = vectors[span2.start : span2.end].mean(axis=0)
                pair_vector = ops.xp.hstack([v1, v2])
                features.append(pair_vector)
                
            X = ops.asarray2f(features)
            classifier = model.get_ref("classifier")
            Y = classifier.predict(X)
            return Y, lambda dY: []

    # Define simple classifier layer
    from thinc.api import Linear, Softmax
    classifier_input_dim = tok2vec.get_dim("nO") * 2 if tok2vec.has_dim("nO") else 96 * 2
    output_dim = nO if nO is not None else 7 # Defaults to 7 relation labels
    
    classifier = chain(
        Relu(nO=64, nI=classifier_input_dim),
        Linear(nO=output_dim, nI=64),
        Softmax()
    )
    
    model = Model(
        "relation_classifier",
        forward,
        layers=[tok2vec, classifier],
        refs={"tok2vec": tok2vec, "classifier": classifier},
        attrs={"labels": []}
    )
    return model
