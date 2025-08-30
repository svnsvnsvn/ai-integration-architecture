"""
Simple RAG System Demo
A basic demonstration of Retrieval-Augmented Generation concepts
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Basic text processing
import re
from collections import Counter

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

class SimpleRAGDatabase:
    """A simplified RAG database using basic text matching"""
    
    def __init__(self, db_path: str = "simple_security_db"):
        self.db_path = db_path
        self.init_database()
        
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
        
    def add_vulnerability(self, vuln: VulnerabilityRecord) -> bool:
        """Add a vulnerability to the database"""
        try:
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
            
        except Exception as e:
            print(f"Error adding vulnerability: {e}")
            return False
    
    def simple_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple text-based search using keyword matching"""
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
                    'relevance_score': matches / len(query_terms)
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        conn.close()
        
        return results[:limit]
    
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
        
        return {
            'total_vulnerabilities': total_vulns,
            'severity_distribution': severity_dist,
            'database_path': self.db_path
        }

def create_sample_data() -> List[VulnerabilityRecord]:
    """Create sample vulnerability data for demonstration"""
    return [
        VulnerabilityRecord(
            cve_id="CVE-2024-0001",
            title="SQL Injection in Web Application Framework",
            description="A critical SQL injection vulnerability allows attackers to execute arbitrary SQL commands through improperly sanitized user input in the authentication module.",
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
            description="A buffer overflow vulnerability in the network service daemon allows remote code execution when processing malformed packets.",
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
            description="Stored XSS vulnerability allows attackers to inject malicious scripts that execute in other users' browsers when viewing user profiles.",
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
            description="An authentication bypass vulnerability allows unauthorized access to protected API endpoints through manipulation of JWT tokens.",
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
            description="A privilege escalation vulnerability allows container processes to escape and gain host system privileges through improper capability handling.",
            severity="CRITICAL",
            cvss_score=9.3,
            published_date="2024-05-20",
            affected_products=["ContainerRuntime v1.8", "ContainerRuntime v1.9"],
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0005"],
            mitigations=["Update container runtime", "Implement proper capability restrictions", "Use security contexts"]
        )
    ]

def demo_rag_system():
    """Demonstrate the Simple RAG system"""
    print("=== Simple RAG System Demo ===")
    print("Initializing security vulnerability database...\n")
    
    # Initialize database
    rag_db = SimpleRAGDatabase()
    
    # Add sample data
    sample_vulns = create_sample_data()
    
    print("Adding sample vulnerability data...")
    for vuln in sample_vulns:
        success = rag_db.add_vulnerability(vuln)
        if success:
            print(f"✓ Added {vuln.cve_id}: {vuln.title}")
        else:
            print(f"✗ Failed to add {vuln.cve_id}")
    
    print(f"\n{'='*60}")
    
    # Show database stats
    stats = rag_db.get_stats()
    print("DATABASE STATISTICS:")
    print(f"Total Vulnerabilities: {stats['total_vulnerabilities']}")
    print("Severity Distribution:")
    for severity, count in stats['severity_distribution'].items():
        print(f"  {severity}: {count}")
    
    print(f"\n{'='*60}")
    
    # Demonstrate search functionality
    test_queries = [
        "SQL injection authentication",
        "buffer overflow network",
        "XSS cross-site scripting",
        "container privilege escalation",
        "API gateway authentication"
    ]
    
    print("SEARCH DEMONSTRATIONS:")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag_db.simple_search(query, limit=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['cve_id']}] {result['title']}")
                print(f"     Severity: {result['severity']} (CVSS: {result['cvss_score']})")
                print(f"     Description: {result['description']}")
                print(f"     Relevance Score: {result['relevance_score']:.2f}")
                if result['mitigations']:
                    print(f"     Mitigations: {', '.join(result['mitigations'][:2])}")
                print()
        else:
            print("  No results found.")
    
    print(f"{'='*60}")
    print("RAG SYSTEM FEATURES DEMONSTRATED:")
    print("1. ✓ Data Ingestion: Structured vulnerability records")
    print("2. ✓ Storage: SQLite database with JSON fields")  
    print("3. ✓ Search: Keyword-based relevance scoring")
    print("4. ✓ Retrieval: Ranked results with metadata")
    print("5. ✓ Security: Content hashing and validation")
    
    print(f"\n{'='*60}")
    print("LEARNING OBJECTIVES:")
    print("• Understanding RAG architecture concepts")
    print("• Text processing and search algorithms")
    print("• Database design for unstructured data")
    print("• Relevance scoring mechanisms")
    print("• Security vulnerability management")
    
    return rag_db

if __name__ == "__main__":
    try:
        demo_rag_system()
        print(f"\nDemo completed successfully!")
        print("Database files created in: simple_security_db/")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
