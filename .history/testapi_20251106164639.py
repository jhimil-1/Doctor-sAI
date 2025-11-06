"""
Test script for GenCare Assistant API
"""
import requests
import json
import uuid
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"


def print_response(response: Dict[Any, Any], title: str):
    """Pretty print API response"""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print('=' * 80)
    print(json.dumps(response, indent=2))
    print('=' * 80)


def test_health_check():
    """Test health endpoint"""
    print("\n🏥 Testing Health Check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print_response(response.json(), "Health Check Result")
    return response.status_code == 200


def test_root():
    """Test root endpoint"""
    print("\n🏠 Testing Root Endpoint...")
    response = requests.get(f"{API_BASE_URL}/")
    print_response(response.json(), "Root Endpoint Result")
    return response.status_code == 200


def test_chat(session_id: str, query: str):
    """Test chat endpoint"""
    print(f"\n💬 Testing Chat: '{query[:50]}...'")
    
    payload = {
        "session_id": session_id,
        "query": query
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_response(response.json(), f"Chat Response (Session: {session_id})")
            return True, response.json()
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(response.text)
            return False, None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, None


def run_conversation_test():
    """Test a multi-turn conversation"""
    print("\n" + "=" * 80)
    print(" 🗣️  MULTI-TURN CONVERSATION TEST")
    print("=" * 80)
    
    # Generate unique session ID
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    print(f"\n📝 Session ID: {session_id}")
    
    # Test queries
    queries = [
        "What is the GenCare system?",
        "What are the main features?",
        "Can you explain more about the security features?",
        "What database does it use?",
        "How do I get started?"
    ]
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n--- Turn {i}/{len(queries)} ---")
        success, response = test_chat(session_id, query)
        results.append(success)
        
        if success and response:
            print(f"\n📊 Response Summary:")
            print(f"   Sources Used: {response.get('sources_used', 0)}")
            print(f"   Answer Length: {len(response.get('answer', ''))} characters")
    
    return all(results)


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" 🚀 GENCARE ASSISTANT API TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Health Check
    results['health'] = test_health_check()
    
    # Test 2: Root Endpoint
    results['root'] = test_root()
    
    # Test 3: Single Chat Query
    print("\n📝 Testing Single Query...")
    session_id = f"single-test-{uuid.uuid4().hex[:8]}"
    success, _ = test_chat(session_id, "What is the purpose of the GenCare system?")
    results['single_chat'] = success
    
    # Test 4: Multi-turn Conversation
    results['conversation'] = run_conversation_test()
    
    # Test 5: Unknown Information
    print("\n🔍 Testing Unknown Information Handling...")
    session_id = f"unknown-test-{uuid.uuid4().hex[:8]}"
    success, response = test_chat(
        session_id,
        "What is the weather forecast for tomorrow?"
    )
    results['unknown_info'] = success
    if response:
        answer = response.get('answer', '').lower()
        handled_correctly = (
            "couldn't find" in answer or 
            "not in the" in answer or
            "documentation" in answer
        )
        print(f"\n✓ Correctly handled unknown info: {handled_correctly}")
        results['unknown_info'] = handled_correctly
    
    # Print Summary
    print("\n" + "=" * 80)
    print(" 📊 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print("=" * 80)
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n🎯 Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        exit(1)