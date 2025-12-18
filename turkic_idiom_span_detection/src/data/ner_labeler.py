from typing import Optional, Dict, List
from utils.morph_matching import MorphMatcher

class IdiomSpanLabeler:
    def label(
        self,
        sentence: str,
        idiom: str,
        lang: str,
        idiom_id: int
    ) -> Optional[Dict]:

        tokens = sentence.split()
        idiom_tokens = idiom.split()
        n = len(idiom_tokens)

        for i in range(len(tokens) - n + 1):
            if all(
                MorphMatcher.match(it, wt)
                for it, wt in zip(idiom_tokens, tokens[i:i+n])
            ):
                labels = ["O"] * len(tokens)
                labels[i] = "B-IDIOM"
                for j in range(i+1, i+n):
                    labels[j] = "I-IDIOM"

                return {
                    "tokens": tokens,
                    "labels": labels,
                    "lang": lang,
                    "idiom_id": idiom_id
                }
        return None
