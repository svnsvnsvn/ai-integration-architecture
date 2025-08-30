"""
Security Audit Script for Weather MCP Server
Demonstrates security considerations and checks
"""

import os
import re
import json
import asyncio
import httpx
from pathlib import Path
from datetime import datetime

class SecurityAuditor:
    def __init__(self):
        self.findings = []
        self.passed = []
    
    def log_finding(self, level: str, message: str):
        """Log a security finding"""
        self.findings.append({
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_pass(self, message: str):
        """Log a passed security check"""
        self.passed.append(message)
    
    def check_env_security(self):
        """Check environment variable security"""
        print("Checking environment security...")
        
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                content = f.read()
                
            # Check for dummy values
            if "your_api_key_here" in content:
                self.log_finding("HIGH", "API key not set in .env file")
            else:
                self.log_pass("API key appears to be configured")
            
            # Check for hardcoded secrets in code
            for py_file in Path(".").glob("*.py"):
                with open(py_file) as f:
                    code = f.read()
                    
                # Look for potential hardcoded secrets
                if re.search(r'api_key\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
                    self.log_finding("CRITICAL", f"Potential hardcoded API key in {py_file}")
                else:
                    self.log_pass(f"No hardcoded secrets found in {py_file}")
        else:
            self.log_finding("HIGH", ".env file not found")
    
    def check_input_validation(self):
        """Check input validation in code"""
        print("Checking input validation...")
        
        server_file = Path("weather_server.py")
        if server_file.exists():
            with open(server_file) as f:
                code = f.read()
            
            # Check for Pydantic models
            if "BaseModel" in code and "Field" in code:
                self.log_pass("Using Pydantic for input validation")
            else:
                self.log_finding("MEDIUM", "No strong input validation framework detected")
            
            # Check for SQL injection protection (not applicable here, but good practice)
            if "SELECT" in code.upper() and "%" in code:
                self.log_finding("HIGH", "Potential SQL injection vulnerability")
            else:
                self.log_pass("No obvious SQL injection risks")
    
    async def check_rate_limiting(self):
        """Test rate limiting functionality"""
        print("Testing rate limiting...")
        
        try:
            # Try to make many requests quickly
            async with httpx.AsyncClient() as client:
                responses = []
                for i in range(12):
                    try:
                        response = await client.post(
                            "http://127.0.0.1:8000/weather",
                            json={"city": "London", "units": "metric"},
                            timeout=5.0
                        )
                        responses.append(response.status_code)
                    except:
                        responses.append(None)
                
                # Check if we got rate limited
                if 429 in responses:
                    self.log_pass("Rate limiting is working")
                else:
                    self.log_finding("MEDIUM", "Rate limiting not triggered in test")
                    
        except Exception as e:
            self.log_finding("LOW", f"Could not test rate limiting: {e}")
    
    def check_error_handling(self):
        """Check error handling security"""
        print("Checking error handling...")
        
        server_file = Path("weather_server.py")
        if server_file.exists():
            with open(server_file) as f:
                code = f.read()
            
            # Check for proper exception handling
            if "HTTPException" in code and "try:" in code:
                self.log_pass("Using structured error handling")
            else:
                self.log_finding("MEDIUM", "Limited error handling detected")
            
            # Check for information disclosure
            if "Internal server error" in code:
                self.log_pass("Generic error messages for internal errors")
            else:
                self.log_finding("LOW", "Consider using generic error messages")
    
    def generate_report(self):
        """Generate security audit report"""
        print("\n" + "="*60)
        print("SECURITY AUDIT REPORT")
        print("="*60)
        
        print(f"\nPASSED CHECKS ({len(self.passed)}):")
        for check in self.passed:
            print(f"   [PASS] {check}")
        
        if self.findings:
            print(f"\nSECURITY FINDINGS ({len(self.findings)}):")
            for finding in self.findings:
                icon = "[CRIT]" if finding["level"] == "CRITICAL" else "[HIGH]" if finding["level"] == "HIGH" else "[MED]" if finding["level"] == "MEDIUM" else "[LOW]"
                print(f"   {icon} {finding['message']}")
        else:
            print("\nNo security findings detected!")
        
        print(f"\nSUMMARY:")
        print(f"   Total checks passed: {len(self.passed)}")
        print(f"   Security findings: {len(self.findings)}")
        
        critical = len([f for f in self.findings if f["level"] == "CRITICAL"])
        high = len([f for f in self.findings if f["level"] == "HIGH"])
        
        if critical > 0:
            print(f"   [CRITICAL] Security Status: CRITICAL - Address {critical} critical issues")
        elif high > 0:
            print(f"   [HIGH] Security Status: NEEDS ATTENTION - Address {high} high-priority issues")
        else:
            print(f"   [GOOD] Security Status: GOOD")

async def main():
    """Run security audit"""
    print("Weather MCP Server Security Audit")
    print("="*50)
    
    auditor = SecurityAuditor()
    
    # Run security checks
    auditor.check_env_security()
    auditor.check_input_validation()
    auditor.check_error_handling()
    
    # Test rate limiting (requires server to be running)
    print("\nTesting live server endpoints...")
    try:
        await auditor.check_rate_limiting()
    except Exception as e:
        print(f"Could not test live server: {e}")
        print("   Make sure the server is running: python weather_server.py")
    
    # Generate report
    auditor.generate_report()
    
    # Save report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "passed": auditor.passed,
        "findings": auditor.findings
    }
    
    with open("security_audit_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nReport saved to: security_audit_report.json")

if __name__ == "__main__":
    asyncio.run(main())
