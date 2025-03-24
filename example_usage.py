import json
import logging
import time
from typing import Dict, Any
from api_client import ApiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_section(title: str) -> None:
    """Print a section header for better readability."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")

def format_response(response: Dict[str, Any], provider: str) -> str:
    """Format API response for display."""
    if "error" in response:
        return f"❌ Error: {response['error']}"
    
    if provider == "anthropic":
        try:
            content = response.get("content", [{}])[0].get("text", "No response")
            return f"✅ Response: {content}"
        except (KeyError, IndexError):
            return f"⚠️ Unexpected response format: {json.dumps(response, indent=2)}"
    
    elif provider == "openai":
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            return f"✅ Response: {content}"
        except (KeyError, IndexError):
            return f"⚠️ Unexpected response format: {json.dumps(response, indent=2)}"
    
    return f"⚠️ Unknown provider: {provider}"

def main() -> None:
    """Example usage of the secure API client."""
    print_section("SECURE API KEY EXAMPLE")
    print("This example demonstrates secure API key management for AI services.")
    
    # Initialize API client
    client = ApiClient()
    
    # Validate configuration
    print_section("SETUP VALIDATION")
    success, validation = client.validate_setup()
    
    print("Configuration validation:")
    print(f"- Anthropic API key: {'✅ Available' if validation['anthropic']['key_available'] else '❌ Missing'}")
    print(f"- OpenAI API key: {'✅ Available' if validation['openai']['key_available'] else '❌ Missing'}")
    
    if not success:
        print("\n⚠️ No API keys configured. Please set up your API keys first:")
        print("1. Copy .env.example to .env and add your keys, or")
        print("2. Copy config.private.toml.example to config.private.toml and add your keys")
        return
    
    # Test message
    test_message = "Tell me a short joke about programming in one sentence."
    
    # Test with Anthropic's Claude
    if validation['anthropic']['key_available']:
        print_section("TESTING ANTHROPIC API")
        print(f"Request: '{test_message}'")
        
        start_time = time.time()
        claude_response = client.call_anthropic_api(test_message)
        elapsed = time.time() - start_time
        
        print(f"Response time: {elapsed:.2f} seconds")
        print(format_response(claude_response, "anthropic"))
    
    # Test with OpenAI's GPT
    if validation['openai']['key_available']:
        print_section("TESTING OPENAI API")
        print(f"Request: '{test_message}'")
        
        start_time = time.time()
        openai_response = client.call_openai_api(test_message)
        elapsed = time.time() - start_time
        
        print(f"Response time: {elapsed:.2f} seconds")
        print(format_response(openai_response, "openai"))
    
    print_section("SECURITY REMINDER")
    print("✅ This example safely loaded API keys without exposing them")
    print("✅ The .gitignore file prevents committing sensitive files")
    print("✅ API keys are never logged or exposed in memory longer than needed")
    print("\nFor production use, consider:")
    print("- Using a dedicated secrets manager")
    print("- Implementing API key rotation")
    print("- Setting up proper access controls")
    
if __name__ == "__main__":
    main()
