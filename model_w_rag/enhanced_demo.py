"""
Enhanced RAG System Demo with Semantic Search
Demonstrates both keyword and semantic search capabilities
"""

import os
import json
import sqlite3
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Basic text processing
import re
from collections import Counter

# Semantic search components
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    SEMANTIC_AVAILABLE = True
    print("Semantic search capabilities loaded")
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Semantic search not available - install sentence-transformers and chromadb")

@dataclass
class VulnerabilityRecord:
    """Security vulnerability record structure"""
    cve_id: str
    title: str
    description: str
    severity: str
    cvss_score: float
    published_date: str
    affected_products: List[str]
    references: List[str]
    mitigations: List[str]

class EnhancedRAGDatabase:
    """RAG database with both keyword and semantic search"""
    
    def __init__(self, db_path: str = "enhanced_security_db", enable_semantic: bool = True):
        self.db_path = db_path
        self.enable_semantic = enable_semantic and SEMANTIC_AVAILABLE
        
        # Initialize keyword search database
        self.init_database()
        
        # Initialize semantic search if available
        if self.enable_semantic:
            self.init_semantic_search()
        
    def init_database(self):
        """Initialize the SQLite database"""
        os.makedirs(self.db_path, exist_ok=True)
        
        conn = sqlite3.connect(f"{self.db_path}/vulnerabilities.db")
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vulnerabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cve_id TEXT UNIQUE,
            title TEXT,
            description TEXT,
            severity TEXT,
            cvss_score REAL,
            published_date TEXT,
            affected_products TEXT,
            refs TEXT,
            mitigations TEXT,
            content_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
    
    def init_semantic_search(self):
        """Initialize semantic search components"""
        try:
            print("Loading semantic search model...")
            # Use a lightweight but effective model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB for vector storage
            self.chroma_client = chromadb.PersistentClient(
                path=f"{self.db_path}/chroma_db"
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="vulnerability_embeddings",
                metadata={"description": "Security vulnerability embeddings"}
            )
            
            print("Semantic search initialized")
            
        except Exception as e:
            print(f"Failed to initialize semantic search: {e}")
            self.enable_semantic = False
    
    def add_vulnerability(self, vuln: VulnerabilityRecord) -> bool:
        """Add a vulnerability to both keyword and semantic databases"""
        try:
            # Add to SQLite (keyword search)
            success = self._add_to_sqlite(vuln)
            
            # Add to vector database (semantic search)
            if self.enable_semantic and success:
                self._add_to_vector_db(vuln)
            
            return success
            
        except Exception as e:
            print(f"Error adding vulnerability: {e}")
            return False
    
    def _add_to_sqlite(self, vuln: VulnerabilityRecord) -> bool:
        """Add vulnerability to SQLite database"""
        conn = sqlite3.connect(f"{self.db_path}/vulnerabilities.db")
        cursor = conn.cursor()
        
        # Create content hash
        content = f"{vuln.title} {vuln.description}"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        cursor.execute("""
        INSERT OR REPLACE INTO vulnerabilities 
        (cve_id, title, description, severity, cvss_score, published_date, 
         affected_products, refs, mitigations, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vuln.cve_id, vuln.title, vuln.description, vuln.severity,
            vuln.cvss_score, vuln.published_date,
            json.dumps(vuln.affected_products),
            json.dumps(vuln.references),
            json.dumps(vuln.mitigations),
            content_hash
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def _add_to_vector_db(self, vuln: VulnerabilityRecord):
        """Add vulnerability to vector database for semantic search"""
        if not self.enable_semantic:
            return
        
        # Create searchable text combining title and description
        searchable_text = f"{vuln.title}. {vuln.description}"
        
        # Generate embedding
        embedding = self.embeddings_model.encode(searchable_text).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[searchable_text],
            metadatas=[{
                "cve_id": vuln.cve_id,
                "title": vuln.title,
                "severity": vuln.severity,
                "cvss_score": vuln.cvss_score,
                "published_date": vuln.published_date
            }],
            ids=[vuln.cve_id]
        )
    
    def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Original keyword-based search"""
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        conn = sqlite3.connect(f"{self.db_path}/vulnerabilities.db")
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT cve_id, title, description, severity, cvss_score, 
               affected_products, mitigations
        FROM vulnerabilities
        """)
        
        results = []
        for row in cursor.fetchall():
            # Calculate simple relevance score
            text_content = f"{row[1]} {row[2]}".lower()
            content_terms = set(re.findall(r'\w+', text_content))
            
            # Simple intersection-based scoring
            matches = len(query_terms.intersection(content_terms))
            
            if matches > 0:
                results.append({
                    'cve_id': row[0],
                    'title': row[1],
                    'description': row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                    'severity': row[3],
                    'cvss_score': row[4],
                    'affected_products': json.loads(row[5]) if row[5] else [],
                    'mitigations': json.loads(row[6]) if row[6] else [],
                    'relevance_score': matches / len(query_terms),
                    'search_type': 'keyword'
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        conn.close()
        
        return results[:limit]
    
    def semantic_search(self, query: str, limit: int = 5, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Semantic search using embeddings"""
        if not self.enable_semantic:
            print("Semantic search not available - falling back to keyword search")
            return self.keyword_search(query, limit)
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode(query).tolist()
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more results to filter by threshold
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            processed_results = []
            for i, distance in enumerate(results['distances'][0]):
                similarity_score = 1 - distance  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold:
                    metadata = results['metadatas'][0][i]
                    document = results['documents'][0][i]
                    
                    processed_results.append({
                        'cve_id': metadata['cve_id'],
                        'title': metadata['title'],
                        'description': document[len(metadata['title']) + 2:],  # Remove title from description
                        'severity': metadata['severity'],
                        'cvss_score': metadata['cvss_score'],
                        'similarity_score': similarity_score,
                        'search_type': 'semantic'
                    })
            
            return processed_results[:limit]
            
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return self.keyword_search(query, limit)
    
    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Combine keyword and semantic search results"""
        
        # Get results from both methods
        keyword_results = self.keyword_search(query, limit)
        semantic_results = self.semantic_search(query, limit) if self.enable_semantic else []
        
        # Combine and deduplicate by CVE ID
        seen_cves = set()
        combined_results = []
        
        # Add semantic results first (usually higher quality)
        for result in semantic_results:
            if result['cve_id'] not in seen_cves:
                result['hybrid_rank'] = len(combined_results) + 1
                combined_results.append(result)
                seen_cves.add(result['cve_id'])
        
        # Add keyword results that weren't found semantically
        for result in keyword_results:
            if result['cve_id'] not in seen_cves:
                result['hybrid_rank'] = len(combined_results) + 1
                combined_results.append(result)
                seen_cves.add(result['cve_id'])
        
        return combined_results[:limit]
    
    def compare_search_methods(self, query: str) -> Dict[str, Any]:
        """Compare all three search methods side by side"""
        
        return {
            'query': query,
            'keyword_results': self.keyword_search(query, 3),
            'semantic_results': self.semantic_search(query, 3) if self.enable_semantic else [],
            'hybrid_results': self.hybrid_search(query, 3),
            'semantic_available': self.enable_semantic
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(f"{self.db_path}/vulnerabilities.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM vulnerabilities")
        total_vulns = cursor.fetchone()[0]
        
        cursor.execute("""
        SELECT severity, COUNT(*) 
        FROM vulnerabilities 
        GROUP BY severity
        """)
        severity_dist = dict(cursor.fetchall())
        
        conn.close()
        
        stats = {
            'total_vulnerabilities': total_vulns,
            'severity_distribution': severity_dist,
            'database_path': self.db_path,
            'semantic_search_enabled': self.enable_semantic
        }
        
        if self.enable_semantic:
            try:
                collection_info = self.collection.count()
                stats['vector_embeddings'] = collection_info
            except:
                stats['vector_embeddings'] = 'unavailable'
        
        return stats

def create_sample_data() -> List[VulnerabilityRecord]:
    """Create enhanced sample vulnerability data"""
    return [
        VulnerabilityRecord(
            cve_id="CVE-2024-0001",
            title="SQL Injection in Web Application Framework",
            description="A critical SQL injection vulnerability allows attackers to execute arbitrary SQL commands through improperly sanitized user input in the authentication module. This vulnerability affects the login system and can lead to complete database compromise.",
            severity="CRITICAL",
            cvss_score=9.8,
            published_date="2024-01-15",
            affected_products=["WebFramework v2.1", "WebFramework v2.2"],
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0001"],
            mitigations=["Update to version 2.3", "Implement parameterized queries", "Enable input validation"]
        ),
        VulnerabilityRecord(
            cve_id="CVE-2024-0002", 
            title="Buffer Overflow in Network Service",
            description="A buffer overflow vulnerability in the network service daemon allows remote code execution when processing malformed packets. Attackers can exploit this by sending specially crafted network requests to execute arbitrary code on the target system.",
            severity="HIGH",
            cvss_score=8.1,
            published_date="2024-02-10",
            affected_products=["NetworkService v1.0", "NetworkService v1.1"],
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0002"],
            mitigations=["Apply security patch", "Implement bounds checking", "Use memory-safe languages"]
        ),
        VulnerabilityRecord(
            cve_id="CVE-2024-0003",
            title="Cross-Site Scripting (XSS) in User Dashboard",
            description="Stored XSS vulnerability allows attackers to inject malicious scripts that execute in other users' browsers when viewing user profiles. This can lead to session hijacking and data theft.",
            severity="MEDIUM",
            cvss_score=6.1,
            published_date="2024-03-05",
            affected_products=["Dashboard v3.0", "Dashboard v3.1", "Dashboard v3.2"],
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0003"],
            mitigations=["Sanitize user input", "Implement Content Security Policy", "Use output encoding"]
        ),
        VulnerabilityRecord(
            cve_id="CVE-2024-0004",
            title="Authentication Bypass in API Gateway",
            description="An authentication bypass vulnerability allows unauthorized access to protected API endpoints through manipulation of JWT tokens. Attackers can forge tokens to gain administrative privileges.",
            severity="HIGH",
            cvss_score=7.5,
            published_date="2024-04-12",
            affected_products=["APIGateway v2.0", "APIGateway v2.1"],
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0004"],
            mitigations=["Update JWT library", "Implement proper token validation", "Use secure signing algorithms"]
        ),
        VulnerabilityRecord(
            cve_id="CVE-2024-0005",
            title="Privilege Escalation in Container Runtime",
            description="A privilege escalation vulnerability allows container processes to escape and gain host system privileges through improper capability handling. This affects containerized environments and can lead to complete system compromise.",
            severity="CRITICAL",
            cvss_score=9.3,
            published_date="2024-05-20",
            affected_products=["ContainerRuntime v1.8", "ContainerRuntime v1.9"],
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0005"],
            mitigations=["Update container runtime", "Implement proper capability restrictions", "Use security contexts"]
        )
    ]

def demo_enhanced_rag_system():
    """Demonstrate the enhanced RAG system with semantic search"""
    print("=== Enhanced RAG System Demo with Semantic Search ===")
    print("Initializing enhanced security vulnerability database...\n")
    
    # Initialize database
    rag_db = EnhancedRAGDatabase()
    
    # Add sample data
    sample_vulns = create_sample_data()
    
    print("Adding sample vulnerability data...")
    for vuln in sample_vulns:
        success = rag_db.add_vulnerability(vuln)
        if success:
            print(f"✓ Added {vuln.cve_id}: {vuln.title}")
        else:
            print(f"✗ Failed to add {vuln.cve_id}")
    
    print(f"\n{'='*70}")
    
    # Show database stats
    stats = rag_db.get_stats()
    print("DATABASE STATISTICS:")
    print(f"Total Vulnerabilities: {stats['total_vulnerabilities']}")
    print("Severity Distribution:")
    for severity, count in stats['severity_distribution'].items():
        print(f"  {severity}: {count}")
    print(f"Semantic Search: {'Enabled' if stats['semantic_search_enabled'] else 'Disabled'}")
    if 'vector_embeddings' in stats:
        print(f"Vector Embeddings: {stats['vector_embeddings']}")
    
    print(f"\n{'='*70}")
    
    # Demonstrate different search methods
    test_queries = [
        "database injection attacks",
        "memory corruption vulnerabilities", 
        "web application security flaws",
        "container security issues"
    ]
    
    print("SEARCH METHOD COMPARISON:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("=" * 50)
        
        comparison = rag_db.compare_search_methods(query)
        
        # Show keyword results
        print("KEYWORD SEARCH:")
        if comparison['keyword_results']:
            for i, result in enumerate(comparison['keyword_results'], 1):
                print(f"  {i}. [{result['cve_id']}] {result['title']}")
                print(f"     Score: {result['relevance_score']:.2f}")
        else:
            print("  No results found")
        
        # Show semantic results  
        if comparison['semantic_available']:
            print("\nSEMANTIC SEARCH:")
            if comparison['semantic_results']:
                for i, result in enumerate(comparison['semantic_results'], 1):
                    print(f"  {i}. [{result['cve_id']}] {result['title']}")
                    print(f"     Similarity: {result['similarity_score']:.2f}")
            else:
                print("  No results found")
        
        # Show hybrid results
        print("\nHYBRID SEARCH:")
        if comparison['hybrid_results']:
            for i, result in enumerate(comparison['hybrid_results'], 1):
                search_type = result.get('search_type', 'unknown')
                score_key = 'similarity_score' if 'similarity_score' in result else 'relevance_score'
                score = result.get(score_key, 0)
                print(f"  {i}. [{result['cve_id']}] {result['title']}")
                print(f"     Method: {search_type}, Score: {score:.2f}")
        else:
            print("  No results found")
    
    print(f"\n{'='*70}")
    print("ENHANCED RAG FEATURES DEMONSTRATED:")
    print("1. ✓ Keyword Search: Term based matching with relevance scoring")
    if stats['semantic_search_enabled']:
        print("2. ✓ Semantic Search: AI-powered understanding of meaning and context")
        print("3. ✓ Hybrid Search: Best of both keyword and semantic approaches")
        print("4. ✓ Vector Embeddings: High-dimensional semantic representations")
    else:
        print("2. Semantic Search: Available but not enabled (install dependencies)")
    print("5. ✓ Comparative Analysis: Side-by-side method comparison")
    print("6. ✓ Enhanced Descriptions: More detailed vulnerability information")
    
    return rag_db

if __name__ == "__main__":
    try:
        demo_enhanced_rag_system()
        print(f"\nEnhanced demo completed successfully!")
        print("Database files created in: enhanced_security_db/")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
