from .text_normalization import TextNormalizer

class MorphMatcher:
    SUFFIXES = [
        "im","ım","um","üm","in","ın","un","ün",
        "ler","lar","nin","nın","dan","den",
        "ga","ka","lik","siz"
    ]

    @staticmethod
    def rough_stem(tok: str) -> str:
        for suf in sorted(MorphMatcher.SUFFIXES, key=len, reverse=True):
            if tok.endswith(suf) and len(tok) > len(suf) + 1:
                return tok[:-len(suf)]
        return tok

    @staticmethod
    def levenshtein(a: str, b: str) -> int:
        if len(a) < len(b):
            return MorphMatcher.levenshtein(b, a)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, c1 in enumerate(a):
            curr = [i + 1]
            for j, c2 in enumerate(b):
                curr.append(min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (c1 != c2)
                ))
            prev = curr
        return prev[-1]

    @classmethod
    def match(cls, tok1: str, tok2: str, max_dist: int = 1) -> bool:
        t1 = cls.rough_stem(TextNormalizer.normalize(tok1))
        t2 = cls.rough_stem(TextNormalizer.normalize(tok2))

        if not t1 or not t2:
            return False
        if t1 == t2:
            return True
        if cls.levenshtein(t1, t2) <= max_dist:
            return True
        if len(t1) >= 3 and t1 in t2:
            return True
        return False
