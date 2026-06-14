# backend/rag.py - Lightweight RAG System (No PyTorch, Memory Efficient)
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class RAGSystem:
    def __init__(self, persist_dir="./data/chromadb"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Collection name
        self.collection_name = "knowledge_base"
        
        # Use ChromaDB's built-in embedding (NO PyTorch, NO sentence-transformers!)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
    
    def add_document(self, text: str, metadata: Dict, doc_id: str):
        """Add a document to the vector database"""
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        return documents
    
    def add_sql_template(self, question_pattern: str, sql: str):
        """Add SQL template to knowledge base"""
        self.add_document(
            text=f"Question: {question_pattern}\nSQL: {sql}",
            metadata={"type": "sql_template", "question": question_pattern},
            doc_id=f"sql_{abs(hash(question_pattern))}"
        )
    
    def add_kpi_definition(self, kpi_name: str, definition: str, formula: str):
        """Add KPI definition to knowledge base"""
        text = f"KPI: {kpi_name}\nDefinition: {definition}\nFormula: {formula}"
        self.add_document(
            text=text,
            metadata={"type": "kpi", "name": kpi_name},
            doc_id=f"kpi_{kpi_name.lower().replace(' ', '_')}"
        )
    
    def add_business_rule(self, rule_name: str, rule_text: str):
        """Add business rule to knowledge base"""
        self.add_document(
            text=rule_text,
            metadata={"type": "business_rule", "name": rule_name},
            doc_id=f"rule_{rule_name.lower().replace(' ', '_')}"
        )
