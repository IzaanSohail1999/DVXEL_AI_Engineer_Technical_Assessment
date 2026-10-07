import asyncio
from .schemas import Evidence

class VectorStoreClient:
    def __init__(self):
        self.docs = {
        "vertex": [
                {
                        "chunk_id": "vx1",
                        "org_id": "vertex",
                        "content": "Refunds are allowed within 6 calendar days of capture.",
                        "metadata": {
                                "keywords": [
                                        "refund",
                                        "returns",
                                        "capture"
                                ],
                                "topic": "refund"
                        },
                        "score": 0.96
                },
                {
                        "chunk_id": "vx2",
                        "org_id": "vertex",
                        "content": "Settlement exports are produced at 00:45 UTC each day.",
                        "metadata": {
                                "keywords": [
                                        "settlement",
                                        "exports",
                                        "daily"
                                ],
                                "topic": "settlement"
                        },
                        "score": 0.84
                },
                {
                        "chunk_id": "vx3",
                        "org_id": "vertex",
                        "content": "Chargeback evidence is reviewed in the dispute center.",
                        "metadata": {
                                "keywords": [
                                        "chargeback",
                                        "evidence",
                                        "dispute"
                                ],
                                "topic": "chargeback"
                        },
                        "score": 0.69
                }
        ],
        "ember": [
                {
                        "chunk_id": "em1",
                        "org_id": "ember",
                        "content": "Refunds are allowed within 12 calendar days of capture.",
                        "metadata": {
                                "keywords": [
                                        "refund",
                                        "returns",
                                        "capture"
                                ],
                                "topic": "refund"
                        },
                        "score": 0.91
                },
                {
                        "chunk_id": "em2",
                        "org_id": "ember",
                        "content": "Settlement exports are produced at 05:15 UTC each day.",
                        "metadata": {
                                "keywords": [
                                        "settlement",
                                        "exports",
                                        "daily"
                                ],
                                "topic": "settlement"
                        },
                        "score": 0.79
                }
        ]
}
        self.fail_next = False
        self.delay_seconds = 0.05

    async def search(self, query: str, tenant_id: str, top_k: int) -> list[Evidence]:
        await asyncio.sleep(self.delay_seconds)
        if self.fail_next:
            self.fail_next = False
            raise TimeoutError("simulated vector dependency timeout")

        # Step 1: keep only the trusted company's notes.
        company_notes = []
        for company_name in self.docs:
            for note in self.docs[company_name]:
                if note["org_id"] == tenant_id:
                    company_notes.append(note)

        # Step 2: turn the question into a list of lowercase words.
        question_words = []
        for word in query.lower().split():
            word = word.strip(".,?!")
            if word != "":
                question_words.append(word)

        # Step 3: score every remaining note by how many question words it contains.
        scored_notes = []
        for note in company_notes:
            # Read keyword list.
            metadata = note.get("metadata") or {}
            keywords = metadata.get("keywords") or []

            # Add keyword to the sentence.
            note_text = note.get("content", "")
            for keyword in keywords:
                note_text = note_text + " " + str(keyword)

            # Split combined text to lowercase
            note_words = []
            for word in note_text.lower().split():
                word = word.strip(".,?!")
                if word != "":
                    note_words.append(word)

            # Add 1 for each question word that also appears in this note.
            match_count = 0
            for word in question_words:
                if word in note_words:
                    match_count = match_count + 1

            # Save the match count, the stored tie-break score, and the note itself.
            scored_notes.append({
                "match_count": match_count,
                "score": note["score"],
                "note": note,
            })

        # Step 4: highest match count first. Same count uses the stored score.
        scored_notes.sort(key=lambda note: (note["match_count"], note["score"]), reverse=True)

        # Step 5: return only the first top_k notes.
        results = []
        for scored_note in scored_notes[:top_k]:
            results.append(Evidence(**scored_note["note"]))
        return results

vector_client = VectorStoreClient()
