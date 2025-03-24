import os
import sys
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from functools import lru_cache
from dotenv import load_dotenv

# Import tomli based on Python version
if sys.version_info >= (3, 11):
    import tomllib as toml_parser
else:
    try:
        import tomli as toml_parser
    except ImportError:
        raise ImportError(
            "Please install tomli package for Python < 3.11: pip install tomli"
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeySecurityError(Exception):
    """Exception raised for API key security issues."""
    pass


class ApiKeyManager:
    """
    Securely manages API keys with multiple loading strategies and validation.
    
    Security features:
    1. Never logs actual API keys
    2. Validates key format before use
    3. Supports multiple loading methods with clear prioritization
    4. Provides clear error messages for troubleshooting
    """
    
    def __init__(self, validate_keys: bool = True) -> None:
        """
        Initialize the API key manager.
        
        Args:
            validate_keys: Whether to validate key format on loading
        """
        self.keys: Dict[str, Optional[str]] = {"anthropic": None, "openai": None}
        self.config: Dict[str, Any] = {}
        self.validate_keys = validate_keys
        
        # Load keys from various sources with fallbacks
        self._load_configuration()
        
    def _load_configuration(self) -> None:
        """Load configuration and API keys from all available sources."""
        # First load public configuration (no sensitive data)
        self._load_public_config()
        
        # Then try loading secrets in order of precedence:
        # 1. Environment variables (highest priority)
        # 2. Private config file
        self._load_from_env()
        self._load_from_private_config()
        
        # Validate loaded keys
        if self.validate_keys:
            self._validate_keys()

    @lru_cache(maxsize=1)
    def _load_public_config(self) -> None:
        """Load public configuration from TOML file."""
        config_path = Path('config.toml')
        if not config_path.exists():
            logger.warning(f"Public configuration file not found: {config_path}")
            return
            
        try:
            with open(config_path, 'rb') as f:
                self.config = toml_parser.load(f)
                logger.info("Loaded public configuration from config.toml")
        except Exception as e:
            logger.error(f"Error loading public configuration: {e}")

    def _load_from_env(self) -> None:
        """Load API keys from environment variables."""
        # Load environment variables from .env file if it exists
        load_dotenv(override=True)
        
        # Try to get API keys from environment variables
        # Use os.environ.get which is safer than direct access
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        openai_key = os.environ.get('OPENAI_API_KEY')
        
        if anthropic_key:
            self.keys['anthropic'] = anthropic_key
            logger.info("Loaded Anthropic API key from environment variables")
            
        if openai_key:
            self.keys['openai'] = openai_key
            logger.info("Loaded OpenAI API key from environment variables")

    def _load_from_private_config(self) -> None:
        """Load API keys from private TOML configuration file."""
        private_config_path = Path('config.private.toml')
        if not private_config_path.exists():
            return
            
        try:
            with open(private_config_path, 'rb') as f:
                private_config = toml_parser.load(f)
            
            # Extract API keys from private config if not already loaded from env vars
            if 'api' in private_config:
                # Anthropic key
                if (self.keys['anthropic'] is None and 
                    'anthropic' in private_config['api'] and 
                    'api_key' in private_config['api']['anthropic']):
                    self.keys['anthropic'] = private_config['api']['anthropic']['api_key']
                    logger.info("Loaded Anthropic API key from private configuration")
                
                # OpenAI key
                if (self.keys['openai'] is None and 
                    'openai' in private_config['api'] and 
                    'api_key' in private_config['api']['openai']):
                    self.keys['openai'] = private_config['api']['openai']['api_key']
                    logger.info("Loaded OpenAI API key from private configuration")
                    
        except Exception as e:
            logger.error(f"Error loading private configuration: {e}")

    def _validate_keys(self) -> None:
        """Validate API key formats and availability."""
        # Check for missing keys
        missing_keys = [name for name, key in self.keys.items() if not key]
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
            logger.warning("Some API calls may fail due to missing keys")
        
        # Validate Anthropic key format if present
        if self.keys['anthropic']:
            if not self.keys['anthropic'].startswith('sk-ant-'):
                logger.warning("Anthropic API key has invalid format")
                
        # Validate OpenAI key format if present
        if self.keys['openai']:
            if not self.keys['openai'].startswith('sk-'):
                logger.warning("OpenAI API key has invalid format")

    def get_key(self, provider: str) -> Optional[str]:
        """
        Get API key for the specified provider.
        
        Args:
            provider: The API provider name ('anthropic' or 'openai')
            
        Returns:
            The API key if available, None otherwise
        """
        return self.keys.get(provider)

    def get_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for the specified provider or all if None.
        
        Args:
            provider: The API provider name or None for full config
            
        Returns:
            Configuration dictionary
        """
        if provider is None:
            return self.config
        
        if 'api' in self.config and provider in self.config['api']:
            return self.config['api'][provider]
        
        return {}

    def mask_key(self, key: Optional[str]) -> str:
        """
        Mask an API key for safe logging.
        
        Args:
            key: The API key to mask
            
        Returns:
            Masked key string
        """
        if not key:
            return "None"
            
        if len(key) <= 8:
            return "****"
            
        return f"{key[:4]}...{key[-4:]}"


class ApiClient:
    """
    Client for securely making API requests to AI providers.
    
    Security features:
    1. Never logs API keys
    2. Validates configuration before requests
    3. Uses separate methods for different providers
    4. Implements proper error handling
    """
    
    def __init__(self, validate_keys: bool = True) -> None:
        """
        Initialize the API client.
        
        Args:
            validate_keys: Whether to validate key format on loading
        """
        self.key_manager = ApiKeyManager(validate_keys=validate_keys)
        
    def call_anthropic_api(
        self, 
        message: str, 
        model: Optional[str] = None,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Call Anthropic's Claude API securely.
        
        Args:
            message: The message to send to the API
            model: The model to use (defaults to configuration)
            max_tokens: Maximum tokens in response
            
        Returns:
            API response as dictionary
        """
        # Get API key and validate
        api_key = self.key_manager.get_key('anthropic')
        if not api_key:
            return {"error": "Anthropic API key is missing"}
        
        # Get configuration
        config = self.key_manager.get_config('anthropic')
        model = model or config.get('default_model', 'claude-3-sonnet-20240229')
        base_url = config.get('base_url', 'https://api.anthropic.com')
        timeout = config.get('request_timeout', 60)
        
        # Set up headers
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Set up request data
        data = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens
        }
        
        # Make request with proper error handling
        try:
            response = requests.post(
                f"{base_url}/v1/messages",
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling Anthropic API (>{timeout}s)")
            return {"error": f"Request timed out after {timeout} seconds"}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error calling Anthropic API: {e}")
            try:
                error_detail = response.json()
                return {"error": f"HTTP {response.status_code}: {error_detail}"}
            except:
                return {"error": f"HTTP {response.status_code}: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic API: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def call_openai_api(
        self, 
        message: str, 
        model: Optional[str] = None,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Call OpenAI's API securely.
        
        Args:
            message: The message to send to the API
            model: The model to use (defaults to configuration)
            max_tokens: Maximum tokens in response
            
        Returns:
            API response as dictionary
        """
        # Get API key and validate
        api_key = self.key_manager.get_key('openai')
        if not api_key:
            return {"error": "OpenAI API key is missing"}
        
        # Get configuration
        config = self.key_manager.get_config('openai')
        model = model or config.get('default_model', 'gpt-4')
        base_url = config.get('base_url', 'https://api.openai.com/v1')
        timeout = config.get('request_timeout', 45)
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Set up request data
        data = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens
        }
        
        # Make request with proper error handling
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling OpenAI API (>{timeout}s)")
            return {"error": f"Request timed out after {timeout} seconds"}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error calling OpenAI API: {e}")
            try:
                error_detail = response.json()
                return {"error": f"HTTP {response.status_code}: {error_detail}"}
            except:
                return {"error": f"HTTP {response.status_code}: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    def validate_setup(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Quick validation to check if the API client is properly configured.
        
        Returns:
            Tuple with (success status, result dictionary)
        """
        result = {
            "anthropic": {
                "key_available": self.key_manager.get_key('anthropic') is not None,
                "config_available": len(self.key_manager.get_config('anthropic')) > 0
            },
            "openai": {
                "key_available": self.key_manager.get_key('openai') is not None,
                "config_available": len(self.key_manager.get_config('openai')) > 0
            }
        }
        
        success = (
            result["anthropic"]["key_available"] or 
            result["openai"]["key_available"]
        )
        
        return success, result
