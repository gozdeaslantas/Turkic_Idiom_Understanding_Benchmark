import re

class TextNormalizer:
    @staticmethod
    def normalize(token: str) -> str:
        token = token.lower()
        token = (token.replace("ı", "i")
                      .replace("ğ", "g")
                      .replace("ş", "s")
                      .replace("ö", "o")
                      .replace("ü", "u")
                      .replace("ç", "c")
                      .replace("â", "a")
                      .replace("ê", "e")
                      .replace("’", "")
                      .replace("'", "")
                      .replace("`", "")
                      .replace("-", ""))
        return re.sub(r"[^a-z0-9]+", "", token)
