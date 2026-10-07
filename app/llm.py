class DeterministicModel:
    async def answer(self, question: str, evidence: list):
        snippets = []
        for item in evidence:
            content = getattr(item, "content", "")
            cleaned = content.strip()
            if cleaned != "":
                snippets.append(cleaned)
        if not snippets:
            return "I do not have enough evidence to answer that from the knowledge base."
        return " ".join(snippets)

model_client = DeterministicModel()
