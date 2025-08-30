"""
Quick test to demonstrate semantic search capabilities
"""

from enhanced_demo import EnhancedRAGDatabase, create_sample_data

def test_semantic_vs_keyword():
    """Compare semantic vs keyword search effectiveness"""
    
    # Initialize database
    rag = EnhancedRAGDatabase()
    
    # Add sample data
    sample_vulns = create_sample_data()
    for vuln in sample_vulns:
        rag.add_vulnerability(vuln)
    
    print("SEMANTIC vs KEYWORD SEARCH COMPARISON")
    print("=" * 60)
    
    # Test queries that demonstrate semantic understanding
    test_cases = [
        {
            "query": "code injection vulnerabilities",
            "description": "Should find SQL injection (semantic understanding)"
        },
        {
            "query": "memory corruption bugs", 
            "description": "Should find buffer overflow (semantic understanding)"
        },
        {
            "query": "login bypass issues",
            "description": "Should find authentication bypass (semantic understanding)"
        },
        {
            "query": "container escape vulnerabilities",
            "description": "Should find privilege escalation in containers"
        }
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\nTest: {description}")
        print(f"Query: '{query}'")
        print("-" * 40)
        
        # Get keyword results
        keyword_results = rag.keyword_search(query, limit=2)
        print("Keyword Search:")
        if keyword_results:
            for result in keyword_results:
                print(f"  • [{result['cve_id']}] {result['title'][:50]}...")
                print(f"    Score: {result['relevance_score']:.3f}")
        else:
            print("  No matches found")
        
        # Get semantic results with lower threshold
        semantic_results = rag.semantic_search(query, limit=2, similarity_threshold=0.2)
        print("\nSemantic Search:")
        if semantic_results:
            for result in semantic_results:
                print(f"  • [{result['cve_id']}] {result['title'][:50]}...")
                print(f"    Similarity: {result['similarity_score']:.3f}")
        else:
            print("  No matches found")
        
        print()

if __name__ == "__main__":
    test_semantic_vs_keyword()
