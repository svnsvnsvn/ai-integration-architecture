"""
Document Processing Pipeline Demo
Demonstrates the LangGraph document processing workflow
"""

import asyncio
import os
from pathlib import Path
from document_pipeline import DocumentPipeline

# Sample documents for testing
SAMPLE_DOCUMENTS = {
    "security_policy.txt": """
COMPANY SECURITY POLICY

1. ACCESS CONTROL
All employees must use strong passwords and enable two-factor authentication.
Access to sensitive systems requires manager approval.

2. DATA PROTECTION
Confidential data must be encrypted at rest and in transit.
Personal information must be handled according to privacy regulations.

3. INCIDENT RESPONSE
Security incidents must be reported within 24 hours.
All incidents will be investigated and documented.

4. NETWORK SECURITY
Firewalls must be configured to block unauthorized access.
VPN connections are required for remote access.

5. COMPLIANCE
All security measures must comply with industry standards.
Regular audits will be conducted to ensure compliance.
""",
    
    "technical_spec.txt": """
SYSTEM ARCHITECTURE SPECIFICATION

1. OVERVIEW
This document describes the architecture of our secure document processing system.

2. COMPONENTS
- Web Interface: FastAPI-based REST API
- Processing Engine: LangGraph workflow orchestration
- Storage: SQLite database with encrypted fields
- Security: Rate limiting, input validation, audit logging

3. SECURITY CONSIDERATIONS
- All inputs are validated and sanitized
- Sensitive data is encrypted before storage
- Access logs are maintained for audit purposes
- Rate limiting prevents abuse

4. DEPLOYMENT
- Container-based deployment using Docker
- Environment variables for configuration
- Automated security scanning in CI/CD pipeline

5. MONITORING
- Application performance monitoring
- Security event logging
- Error tracking and alerting
""",

    "meeting_notes.txt": """
SECURITY TEAM MEETING NOTES
Date: August 29, 2025

ATTENDEES:
- Security Manager
- Development Team Lead
- AI Security Intern

AGENDA:
1. Review current security posture
2. Discuss AI security implementation
3. Plan upcoming security training

DISCUSSION POINTS:
- AI model security is becoming increasingly important
- Need to implement proper input validation for AI systems
- Document processing pipeline should include security checks
- Rate limiting and monitoring are essential

ACTION ITEMS:
- Implement security audit for AI systems
- Create security guidelines for AI development
- Schedule security training for development team
- Review and update incident response procedures

NEXT MEETING: September 5, 2025
"""
}

def create_sample_documents():
    """Create sample documents for testing"""
    print("Creating sample documents...")
    
    docs_dir = Path("sample_docs")
    docs_dir.mkdir(exist_ok=True)
    
    created_files = []
    
    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = docs_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        created_files.append(str(file_path))
        print(f"   Created: {filename}")
    
    return created_files

async def demo_pipeline():
    """Demonstrate the document processing pipeline"""
    print("Document Processing Pipeline Demo")
    print("="*50)
    
    # Create sample documents
    sample_files = create_sample_documents()
    
    # Initialize pipeline
    pipeline = DocumentPipeline()
    
    print(f"\nProcessing {len(sample_files)} documents...")
    print("="*50)
    
    results = []
    
    # Process each document
    for file_path in sample_files:
        print(f"\nProcessing: {Path(file_path).name}")
        print("-" * 40)
        
        try:
            result = await pipeline.process_document(file_path)
            results.append(result)
            
            if result["status"] == "completed":
                print(f"Successfully processed: {result['document_id']}")
            else:
                print(f"Processing failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Exception occurred: {e}")
            results.append({"status": "exception", "error": str(e), "file_path": file_path})
    
    # Display results summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    successful = len([r for r in results if r["status"] == "completed"])
    failed = len([r for r in results if r["status"] != "completed"])
    
    print(f"Total documents: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Display processed documents
    if successful > 0:
        print(f"\nPROCESSED DOCUMENTS:")
        print("-" * 30)
        
        docs = pipeline.list_processed_documents()
        for doc in docs:
            status_icon = "[UNSAFE]" if not doc["safe_content"] else "[SAFE]"
            print(f"{status_icon} {doc['document_id'][:12]}... - {Path(doc['file_path']).name}")
            print(f"   Type: {doc['file_type'].upper()}")
            print(f"   Summary: {doc['summary'][:80]}...")
            print(f"   Processed: {doc['processed_at']}")
            print()
    
    # Demonstrate document retrieval
    if results and results[0]["status"] == "completed":
        print("DETAILED DOCUMENT VIEW:")
        print("-" * 30)
        
        doc_id = results[0]["document_id"]
        detailed_doc = pipeline.get_document_details(doc_id)
        
        if detailed_doc:
            print(f"Document ID: {detailed_doc.document_id}")
            print(f"File: {Path(detailed_doc.file_path).name}")
            print(f"Content Hash: {detailed_doc.content_hash[:16]}...")
            print(f"Chunks: {detailed_doc.chunk_count}")
            print(f"Safe Content: {detailed_doc.safe_content}")
            print(f"Summary: {detailed_doc.summary}")
            print(f"Key Points:")
            for i, point in enumerate(detailed_doc.key_points, 1):
                print(f"  {i}. {point}")
    
    print("\nDemo completed!")
    return results

def demo_security_features():
    """Demonstrate security features of the pipeline"""
    print("\nSECURITY FEATURES DEMONSTRATION")
    print("="*50)
    
    print("1. Input Validation:")
    print("   - File type validation (PDF, DOCX, TXT only)")
    print("   - File existence checks")
    print("   - Content size limits")
    
    print("\n2. Content Safety Checks:")
    print("   - Scans for sensitive information")
    print("   - Flags documents with potential secrets")
    print("   - Generates content hashes for integrity")
    
    print("\n3. Error Handling:")
    print("   - Graceful error recovery")
    print("   - Detailed error logging")
    print("   - State preservation during failures")
    
    print("\n4. Data Security:")
    print("   - Content hashing for integrity verification")
    print("   - Structured data storage")
    print("   - Audit trail maintenance")
    
    print("\n5. Workflow Security:")
    print("   - State isolation between documents")
    print("   - Conditional processing based on safety checks")
    print("   - Comprehensive logging at each step")

async def main():
    """Main demo function"""
    try:
        await demo_pipeline()
        demo_security_features()
        
        print("\nFILES CREATED:")
        print("   - sample_docs/ (directory with test documents)")
        print("   - documents.db (SQLite database)")
        
        # Check if error log was actually created
        error_log_path = Path("error_log.json")
        if error_log_path.exists():
            print("   - error_log.json (error log file)")
        else:
            print("   - No error log (no errors occurred)")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
