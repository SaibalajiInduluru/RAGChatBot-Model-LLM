import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import requests


class RAGChatbot:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def load_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        chunks = []
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < 500:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        for i, chunk in enumerate(chunks):
            embedding = self.encoder.encode([chunk])[0].tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"chunk_{i}"]
            )

    def ask(self, query):
        query_embedding = self.encoder.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        context = "\n".join(results['documents'][0])

        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        payload = {
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload)
        result = response.json()
        if 'response' in result:
            return result['response']
        else:
            return f"Error: {result}"


bot = RAGChatbot()
bot.load_pdf("balaji.pdf")

while True:
    question = input("Question: ")
    if question.lower() == 'quit':
        break
    print(bot.ask(question))
